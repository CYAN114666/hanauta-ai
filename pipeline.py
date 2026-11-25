import math
import os
import subprocess
import uuid

import librosa
import pretty_midi
from pydub import AudioSegment
from basic_pitch.inference import predict, Model
from basic_pitch import ICASSP_2022_MODEL_PATH

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOUNDFONT = os.environ.get(
    "SOUNDFONT_PATH",
    os.path.join(BASE_DIR, "MuseScore_General.sf2"),
)
WORK_DIR = os.path.join(BASE_DIR, "work")

# Basic Pitchのモデルは最初に1回だけ読み込む（毎回読み込むと遅いため）
model = Model(ICASSP_2022_MODEL_PATH)

# ---- 音程のある楽器（Basic Pitchで音程を読み取る） ----
PROGRAMS = {
    "guitar": 24,  # アコースティックギター（ナイロン弦）
    "bass": 33,    # エレキベース（指弾き）
    "piano": 0,    # アコースティックピアノ
}

# ---- 音域の住み分け（歌の高さを、1オクターブ単位でずらして、各楽器の中心に近づける） ----
TARGET_CENTER = {  # 目標の中心の高さ（MIDI番号）。音域の帯が重ならないように決めている
    "bass": 38,    # D2（音域の帯: E1〜C3 = 28〜48）
    "guitar": 59,  # B3（音域の帯: E3〜G4 = 52〜67）
    "piano": 81,   # A5（音域の帯: C5〜G6 = 72〜91）
}
TRANSPOSE = {      # オクターブ寄せのあとに、さらに半音単位でずらす量（ふつうは0）
    "bass": 0,
    "guitar": 0,
    "piano": 0,
}

# ---- 検出した音の後処理（ここに書いた楽器だけに効く。ベースは入れていない） ----
CLEAN_PARTS = ["guitar", "piano"]
MERGE_GAP_SEC = 0.1                # 同じ高さの音の間がこれ以下なら、1つの音につなげる（秒）
HARMONIC_INTERVALS = (12, 19, 24)  # 倍音になりやすい、元の音からの距離（半音）
ABSORB_SHORT_SEC = 0.25            # これより短い音は、前後の近い高さの音に吸収する（秒）
ABSORB_MAX_GAP_SEC = 0.15          # 吸収するときに許す、音と音のすき間（秒）
ABSORB_MAX_SEMITONES = 2           # 吸収するときに許す、高さの違い（半音）

# ---- 音が伸びるのを抑える設定（ここに書いた楽器だけに効く。ベースは入れていない） ----
MAX_NOTE_SEC = {   # 1音の最大の長さ（秒）。反響などで伸びた音を、この長さで切る
    "guitar": 0.7,
    "piano": 0.8,
}
MIN_GAP_SEC = 0.1  # 次の音との間がこれより短いときは、重なりを整えない（秒）
DRY_PARTS = ["guitar", "piano"]  # リバーブとコーラスを切って、すっきりした音にする楽器

# ---- ドラム（リズムだけ読み取って、決まったドラム音を鳴らす） ----
DRUM_NOTES = {
    "kick": 36,    # バスドラム（ドン）
    "snare": 38,   # スネア（タン）
    "hihat": 42,   # クローズドハイハット（チッ）
}
ONSET_DELTA = 0.1       # 大きいほど、はっきりした音だけを拾う
ONSET_MIN_GAP = 0.1     # 同じ音を二重に拾わないための最小の間隔（秒）
ONSET_SKIP_START = 0.1  # 録音の最初のごく短い間は、ノイズを誤って拾いやすいので無視する（秒）

MIN_NOTE_SEC = 0.02     # 編集で音の長さを縮めても、これより短くはしない（秒）

SUPPORTED_PARTS = list(PROGRAMS) + list(DRUM_NOTES)


def note_name(pitch):
    """MIDI番号を音名にする（60 → C4）"""
    return pretty_midi.note_number_to_name(int(pitch))


def print_detected_notes(midi_data, label):
    """音を1音ずつ一覧にして表示する（開始・長さ・音名・MIDI番号）"""
    notes = sorted(
        (n for inst in midi_data.instruments for n in inst.notes),
        key=lambda n: (n.start, n.pitch),
    )
    print(f"--- 検出した音の一覧（{label}） ---")
    print("開始(秒) 長さ(秒) 音名 MIDI番号")
    for n in notes:
        print(f"{n.start:.2f} {n.end - n.start:.2f} {note_name(n.pitch)} {n.pitch}")


def merge_same_pitch(notes, max_gap=MERGE_GAP_SEC):
    """同じ高さの音が、ほぼ隙間なく続いているときは、1つの音につなげる"""
    merged = []
    for n in sorted(notes, key=lambda n: (n.pitch, n.start)):
        last = merged[-1] if merged else None
        if last is not None and last.pitch == n.pitch and n.start - last.end <= max_gap:
            last.end = max(last.end, n.end)
        else:
            merged.append(n)
    return sorted(merged, key=lambda n: (n.start, n.pitch))


def remove_harmonics(notes, tolerance=1):
    """同時に鳴っている、12・19・24半音上の音は、倍音の誤検出とみなして取り除く"""
    kept = []
    for n in notes:
        is_harmonic = False
        for low in notes:
            if low is n or low.pitch >= n.pitch:
                continue
            overlap = min(n.end, low.end) - max(n.start, low.start)
            if overlap < 0.5 * (n.end - n.start):
                continue
            diff = n.pitch - low.pitch
            if any(abs(diff - h) <= tolerance for h in HARMONIC_INTERVALS):
                is_harmonic = True
                break
        if not is_harmonic:
            kept.append(n)
    return kept


def absorb_short_notes(notes):
    """短い音（声のゆらぎで生まれた断片）を、前後の近い高さの音に吸収する"""
    notes = sorted(notes, key=lambda n: n.start)
    while True:
        for i, n in enumerate(notes):
            if n.end - n.start >= ABSORB_SHORT_SEC:
                continue
            best = None
            for j in (i - 1, i + 1):
                if not 0 <= j < len(notes):
                    continue
                m = notes[j]
                gap = n.start - m.end if j < i else m.start - n.end
                if gap <= ABSORB_MAX_GAP_SEC and abs(m.pitch - n.pitch) <= ABSORB_MAX_SEMITONES:
                    if best is None or (m.end - m.start) > (best.end - best.start):
                        best = m
            if best is not None:
                best.start = min(best.start, n.start)
                best.end = max(best.end, n.end)
                notes.remove(n)
                break
        else:
            return notes


def clean_notes(notes):
    """後処理：倍音を取り除く → 同じ高さをつなげる → 短い断片を吸収 → もう一度つなげる"""
    notes = remove_harmonics(notes)
    notes = merge_same_pitch(notes)
    notes = absorb_short_notes(notes)
    return merge_same_pitch(notes)


def center_pitch(notes):
    """音の高さの中心（音の長さで重みづけした中央値）"""
    pairs = sorted((n.pitch, n.end - n.start) for n in notes)
    half = sum(d for _, d in pairs) / 2
    total = 0.0
    for pitch, d in pairs:
        total += d
        if total >= half:
            return pitch
    return pairs[-1][0]


def octave_shift(notes, target_center):
    """音の高さの中心が、目標の中心に一番近くなるような、1オクターブ単位のずらし量（半音）を返す"""
    if not notes:
        return 0
    center = center_pitch(notes)
    return math.floor((target_center - center) / 12 + 0.5) * 12  # ちょうど真ん中なら、高いほうにする


def shorten_notes(notes, max_sec):
    """反響で伸びた音を整える：次の音が始まる前に止めて、長さの上限も付ける"""
    notes = sorted(notes, key=lambda n: n.start)
    for i, note in enumerate(notes):
        if i + 1 < len(notes) and notes[i + 1].start - note.start >= MIN_GAP_SEC:
            note.end = min(note.end, notes[i + 1].start)
        note.end = min(note.end, note.start + max_sec)


def pitched_to_midi(wav_path, part_type):
    """鼻歌の音程を読み取って、ギター・ベース・ピアノ用のMIDIにする"""
    _, midi_data, _ = predict(wav_path, model)
    note_count = sum(len(inst.notes) for inst in midi_data.instruments)
    print(f"検出した音の数: {note_count}")
    print_detected_notes(midi_data, part_type)  # 検出そのままの結果

    for inst in midi_data.instruments:
        inst.program = PROGRAMS[part_type]
        inst.pitch_bends = []  # 声の細かい音程のゆらぎを消す
        if part_type in CLEAN_PARTS:
            inst.notes = clean_notes(inst.notes)
    if part_type in CLEAN_PARTS:
        print_detected_notes(midi_data, f"{part_type}・後処理のあと")

    for inst in midi_data.instruments:
        if not inst.notes:
            continue
        # 音域の住み分け：1オクターブ単位でずらして、この楽器の中心に近づける
        center = center_pitch(inst.notes)
        shift = octave_shift(inst.notes, TARGET_CENTER[part_type]) + TRANSPOSE[part_type]
        for note in inst.notes:
            note.pitch = max(0, min(127, note.pitch + shift))
        low = min(n.pitch for n in inst.notes)
        high = max(n.pitch for n in inst.notes)
        print(
            f"音域の調整（{part_type}）: {shift:+d}半音"
            f"（中心 {note_name(center)} → {note_name(center + shift)}、"
            f"目標 {note_name(TARGET_CENTER[part_type])}、"
            f"全体の音域 {note_name(low)}〜{note_name(high)}）"
        )
        if part_type in MAX_NOTE_SEC:
            shorten_notes(inst.notes, MAX_NOTE_SEC[part_type])
    return midi_data


def rhythm_to_midi(wav_path, part_type):
    """鼻歌のリズム（音の立ち上がり）だけを読み取って、決まったドラム音のMIDIにする"""
    y, sr = librosa.load(wav_path, sr=22050, mono=True)
    hop_length = 512
    envelope = librosa.onset.onset_strength(
        y=y, sr=sr, hop_length=hop_length, pad_mode="reflect"
    )
    onset_times = librosa.onset.onset_detect(
        onset_envelope=envelope,
        sr=sr,
        hop_length=hop_length,
        units="time",
        delta=ONSET_DELTA,
        wait=int(ONSET_MIN_GAP * sr / hop_length),
    )
    onset_times = [t for t in onset_times if t >= ONSET_SKIP_START]
    print(f"検出したリズムの数: {len(onset_times)}")
    if not onset_times:
        raise ValueError("リズムを検出できませんでした。もう少し大きな声で録音してみてください。")

    midi_data = pretty_midi.PrettyMIDI()
    drums = pretty_midi.Instrument(program=0, is_drum=True)
    for t in onset_times:
        drums.notes.append(
            pretty_midi.Note(
                velocity=100,
                pitch=DRUM_NOTES[part_type],
                start=float(t),
                end=float(t) + 0.1,  # ドラムは一瞬だけ鳴らす
            )
        )
    midi_data.instruments.append(drums)
    return midi_data


def midi_to_notes(midi_data):
    """MIDIのデータを、編集しやすいノート（音1つ分）の一覧にする"""
    notes = []
    for inst in midi_data.instruments:
        for n in inst.notes:
            notes.append({
                "start": round(float(n.start), 3),   # 開始（秒）
                "end": round(float(n.end), 3),       # 終わり（秒）
                "pitch": int(n.pitch),               # 音の高さ（MIDI番号）
                "name": note_name(n.pitch),          # 音名（見るための情報。音には影響しない）
                "velocity": int(n.velocity),         # 音の強さ
            })
    notes.sort(key=lambda x: (x["start"], x["pitch"]))
    return notes


def analyze_audio(input_path, part_type):
    """鼻歌の音声ファイルを、編集できるノートの一覧にして返す（音はまだ作らない）"""
    os.makedirs(WORK_DIR, exist_ok=True)
    job = uuid.uuid4().hex[:8]  # ファイル名がかぶらないようにする目印

    # 1. どんな形式の音声でもwavにそろえる（ffmpegを使う）
    wav_in = os.path.join(WORK_DIR, f"{job}_in.wav")
    AudioSegment.from_file(input_path).export(wav_in, format="wav")

    # 2. 音声 → ノート（楽器によって読み取り方が違う）
    if part_type in DRUM_NOTES:
        midi_data = rhythm_to_midi(wav_in, part_type)   # ドラム：リズムだけ
    else:
        midi_data = pitched_to_midi(wav_in, part_type)  # ギター・ベース・ピアノ：音程
    return midi_to_notes(midi_data)


def midi_to_mp3(midi_data, part_type, job):
    """MIDIのデータを、FluidSynthで楽器の音にして、mp3にして、mp3のパスを返す"""
    midi_path = os.path.join(WORK_DIR, f"{job}.mid")
    midi_data.write(midi_path)

    # FluidSynthでMIDI → wav（オプションは必ずファイル名より前に書く）
    wav_out = os.path.join(WORK_DIR, f"{job}_out.wav")
    command = ["fluidsynth", "-ni", "-F", wav_out, "-r", "44100"]
    if part_type in DRY_PARTS:
        command += ["-R", "0", "-C", "0"]  # リバーブとコーラスを切って、すっきりさせる
    command += [SOUNDFONT, midi_path]
    result = subprocess.run(command, capture_output=True, text=True)
    if not os.path.exists(wav_out):
        raise RuntimeError(f"FluidSynthの変換に失敗: {result.stderr}")

    # wav → mp3
    mp3_path = os.path.join(WORK_DIR, f"{job}.mp3")
    AudioSegment.from_wav(wav_out).export(mp3_path, format="mp3")
    return mp3_path


def render_notes(notes, part_type, sustain_scale=1.0, shift_sec=0.0):
    """ノートの一覧から、指定した楽器の音のmp3を作って、mp3のパスを返す。
    渡されたノートを、そのまま鳴らす（後処理や音域の調整は、ここではかけない）。
    sustain_scale: 全部の音の長さにかける倍率 / shift_sec: 全部の音をずらす秒数"""
    if not notes:
        raise ValueError("ノートが1つもありません。")
    os.makedirs(WORK_DIR, exist_ok=True)
    job = uuid.uuid4().hex[:8]

    midi_data = pretty_midi.PrettyMIDI()
    if part_type in DRUM_NOTES:
        inst = pretty_midi.Instrument(program=0, is_drum=True)
    else:
        inst = pretty_midi.Instrument(program=PROGRAMS[part_type])
    for n in notes:
        start = max(0.0, n["start"] + shift_sec)
        length = max(MIN_NOTE_SEC, (n["end"] - n["start"]) * sustain_scale)
        inst.notes.append(
            pretty_midi.Note(
                velocity=n.get("velocity", 100),
                pitch=n["pitch"],
                start=start,
                end=start + length,
            )
        )
    midi_data.instruments.append(inst)
    return midi_to_mp3(midi_data, part_type, job)


def humming_to_instrument(input_path, part_type):
    """鼻歌の音声ファイルを、指定した楽器の音のmp3に変換して、mp3のパスを返す（分析 → 音づくり をまとめたもの）"""
    notes = analyze_audio(input_path, part_type)
    return render_notes(notes, part_type)