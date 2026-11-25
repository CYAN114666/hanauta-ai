import os
import shutil
import tempfile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from pipeline import SUPPORTED_PARTS, analyze_audio, humming_to_instrument, render_notes

app = FastAPI()
@app.get("/health")
def health():
    return {"status": "ok"}


class Note(BaseModel):
    start: float = Field(ge=0, description="開始（秒）")
    end: float = Field(gt=0, description="終わり（秒）。startより後にする")
    pitch: int = Field(ge=0, le=127, description="音の高さ（MIDI番号。60 = C4）")
    velocity: int = Field(default=100, ge=1, le=127, description="音の強さ")


class RenderRequest(BaseModel):
    part_type: str
    notes: list[Note]
    sustain_scale: float = Field(default=1.0, gt=0, le=10, description="全部の音の長さにかける倍率")
    shift_sec: float = Field(default=0.0, ge=-10, le=10, description="全部の音をずらす秒数")


def check_part_type(part_type: str):
    """対応していないpart_typeは、処理を始める前にエラーで返す"""
    if part_type not in SUPPORTED_PARTS:
        raise HTTPException(
            status_code=400,
            detail=f"part_type は {SUPPORTED_PARTS} のどれかを指定してください",
        )


def save_upload(file: UploadFile) -> str:
    """アップロードされた音声を一時ファイルに保存して、そのパスを返す（拡張子は元のまま）"""
    suffix = os.path.splitext(file.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        return tmp.name


@app.post("/convert")
def convert_audio(
    file: UploadFile = File(...),
    part_type: str = Form(...),
):
    """鼻歌 → 楽器の音のmp3（分析と音づくりを、まとめて行う）"""
    check_part_type(part_type)
    tmp_path = save_upload(file)
    try:
        mp3_path = humming_to_instrument(tmp_path, part_type)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        os.remove(tmp_path)  # 一時ファイルは変換後に消す
    return FileResponse(mp3_path, media_type="audio/mpeg", filename="converted.mp3")


@app.post("/analyze")
def analyze(
    file: UploadFile = File(...),
    part_type: str = Form(...),
):
    """鼻歌 → 編集できるノートの一覧（JSON）。音はまだ作らない"""
    check_part_type(part_type)
    tmp_path = save_upload(file)
    try:
        notes = analyze_audio(tmp_path, part_type)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        os.remove(tmp_path)
    return {"part_type": part_type, "notes": notes}


@app.post("/render")
def render(request: RenderRequest):
    """ノートの一覧（JSON） → 楽器の音のmp3。渡されたノートを、そのまま鳴らす"""
    check_part_type(request.part_type)
    for i, n in enumerate(request.notes):
        if n.end <= n.start:
            raise HTTPException(
                status_code=422,
                detail=f"{i + 1}番目のノートの end（終わり）が、start（開始）より後になっていません",
            )
    try:
        mp3_path = render_notes(
            [n.model_dump() for n in request.notes],
            request.part_type,
            sustain_scale=request.sustain_scale,
            shift_sec=request.shift_sec,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return FileResponse(mp3_path, media_type="audio/mpeg", filename="rendered.mp3")