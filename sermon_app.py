import csv
import html
import json
import os
import shutil
import subprocess
import sys

# ปลดล็อกให้ Windows มองเห็นไฟล์ cublas64_12.dll และ cudnn จาก pip อัตโนมัติ
if sys.platform == "win32":
    for site_path in sys.path:
        nvidia_dir = os.path.join(site_path, "nvidia")
        if os.path.isdir(nvidia_dir):
            for root, dirs, _ in os.walk(nvidia_dir):
                if "bin" in dirs:
                    bin_path = os.path.join(root, "bin")
                    try:
                        os.add_dll_directory(bin_path)
                    except Exception:
                        pass
                    if bin_path not in os.environ.get("PATH", ""):
                        os.environ["PATH"] = bin_path + os.pathsep + os.environ.get("PATH", "")

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTableWidget, QTableWidgetItem, QFileDialog,
    QLabel, QHeaderView, QMessageBox, QProgressBar, QComboBox,
    QDialog, QListWidget, QListWidgetItem, QLineEdit, QColorDialog,
    QSplitter, QGroupBox, QRadioButton, QButtonGroup, QScrollArea,
    QFrame, QAbstractItemView, QSlider
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QUrl, QEvent
from PyQt6.QtGui import QColor, QBrush, QFont
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput

try:
    from faster_whisper import WhisperModel, BatchedInferencePipeline
except ImportError:
    from faster_whisper import WhisperModel
    BatchedInferencePipeline = None


# โฟลเดอร์ถาวรสำหรับเก็บตั้งค่าสีและคลังคำศัพท์ (อัปเดตเวอร์ชันใหม่แล้วค่าไม่หาย)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(os.path.expanduser("~"), "Documents", "SermonTranscriberConfig")
os.makedirs(CONFIG_DIR, exist_ok=True)

CONFIG_FILE = os.path.join(CONFIG_DIR, "sermon_categories.json")
GLOSSARY_FILE = os.path.join(CONFIG_DIR, "sermon_glossary.json")

# ย้ายไฟล์ตั้งค่าเดิมจากโฟลเดอร์โปรแกรม (ถ้ามี) มาไว้ในโฟลเดอร์ถาวรอัตโนมัติ
for fname, target in [
    ("sermon_categories.json", CONFIG_FILE),
    ("sermon_glossary.json", GLOSSARY_FILE),
]:
    legacy_path = os.path.join(BASE_DIR, fname)
    if not os.path.exists(target) and os.path.exists(legacy_path):
        try:
            shutil.copy2(legacy_path, target)
        except Exception:
            pass

DEFAULT_CATEGORIES = [
    {"id": "cat_1", "name": "เนื้อเรื่อง", "color": ""},
    {"id": "cat_2", "name": "คำถาม", "color": "#FFF9C4"},
    {"id": "cat_3", "name": "คำตอบ", "color": "#E1F5FE"},
]

DEFAULT_GLOSSARY = [
    "ภุมมเทวา",
    "อากาสเทวา",
    "เคสสตั๊ดดี้",
    "อนุบาลฝันในฝันวิทยา",
    "สัมมาอะระหัง",
    "วัดพระธรรมกาย",
    "มหากาลทาน",
    "วิบากกรรม",
    "อโหสิกรรม",
    "หลวงปู่สด",
    "ดาวดึงส์",
    "ดุสิตบุรี",
]

GPU_PROFILES = [
    {
        "label": "8GB — เร็วสุดแนะนำ (int8_float16, Batch 8)",
        "compute_type": "int8_float16",
        "batch_size": 8,
    },
    {
        "label": "8GB — แม่นยำเต็ม 100% (float16, Batch 4)",
        "compute_type": "float16",
        "batch_size": 4,
    },
    {
        "label": "16GB — เต็มกำลัง RTX 5080 (float16, Batch 16)",
        "compute_type": "float16",
        "batch_size": 16,
    },
    {
        "label": "โหมดปกติ — ประหยัด VRAM (float16, Batch 1)",
        "compute_type": "float16",
        "batch_size": 1,
    },
]


def load_categories():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    return data
        except Exception:
            pass
    return [dict(c) for c in DEFAULT_CATEGORIES]


def save_categories(categories):
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(categories, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def load_glossary():
    if os.path.exists(GLOSSARY_FILE):
        try:
            with open(GLOSSARY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return [str(w).strip() for w in data if str(w).strip()]
        except Exception:
            pass
    return list(DEFAULT_GLOSSARY)


def save_glossary(words):
    try:
        with open(GLOSSARY_FILE, "w", encoding="utf-8") as f:
            json.dump(words, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def get_ffmpeg_executable():
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.exists(exe):
            return exe
    except Exception:
        pass
    return shutil.which("ffmpeg")


def parse_time_str(time_str):
    s = time_str.strip()
    if not s:
        return None
    parts = s.split(":")
    try:
        if len(parts) == 1:
            val = float(parts[0])
        elif len(parts) == 2:
            val = int(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3:
            val = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        else:
            return None
        return max(0.0, val)
    except ValueError:
        return None


def format_hms(seconds, show_decimal=False):
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if show_decimal:
        return f"{h:02d}:{m:02d}:{s:04.1f}"
    return f"{h:02d}:{m:02d}:{int(round(s)):02d}"


def format_srt_time(seconds):
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms >= 1000:
        ms = 999
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def get_contrast_text_color(hex_color):
    if not hex_color:
        return QColor("#1F2937")
    c = QColor(hex_color)
    if not c.isValid():
        return QColor("#1F2937")
    luminance = (0.299 * c.red() + 0.587 * c.green() + 0.114 * c.blue())
    return QColor("#1A1A1A") if luminance > 128 else QColor("#FFFFFF")


# 1. หน้าต่างจัดการชนิดเนื้อเรื่องและสี
class CategoryManagerDialog(QDialog):
    def __init__(self, categories, parent=None):
        super().__init__(parent)
        self.setWindowTitle("จัดการชนิดเนื้อเรื่องและสีไฮไลต์")
        self.resize(520, 440)
        self.categories = [dict(c) for c in categories]
        self.next_id_num = len(self.categories) + 20
        self.init_ui()
        self.refresh_list()

    def init_ui(self):
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("รายการชนิดเนื้อเรื่อง (คลิกเลือกเพื่อแก้ไขชื่อ เปลี่ยนสี หรือลบ):"))
        self.list_widget = QListWidget()
        self.list_widget.currentRowChanged.connect(self.on_select_row)
        layout.addWidget(self.list_widget)

        edit_row = QHBoxLayout()
        self.txt_name = QLineEdit()
        self.txt_name.setPlaceholderText("พิมพ์ชื่อชนิดเนื้อเรื่อง...")
        self.btn_color = QPushButton("เลือกสี")
        self.btn_color.clicked.connect(self.pick_color)
        self.btn_clear_color = QPushButton("ใช้สีเดิม")
        self.btn_clear_color.clicked.connect(self.clear_color)

        edit_row.addWidget(self.txt_name)
        edit_row.addWidget(self.btn_color)
        edit_row.addWidget(self.btn_clear_color)
        layout.addLayout(edit_row)

        action_row = QHBoxLayout()
        self.btn_add = QPushButton("+ เพิ่มชนิดใหม่")
        self.btn_add.clicked.connect(self.add_category)
        self.btn_update = QPushButton("บันทึกชื่อที่แก้")
        self.btn_update.clicked.connect(self.update_category_name)
        self.btn_delete = QPushButton("ลบชนิดที่เลือก")
        self.btn_delete.clicked.connect(self.delete_category)

        action_row.addWidget(self.btn_add)
        action_row.addWidget(self.btn_update)
        action_row.addWidget(self.btn_delete)
        layout.addLayout(action_row)

        bottom_row = QHBoxLayout()
        bottom_row.addStretch()
        btn_ok = QPushButton("ตกลงและนำไปใช้")
        btn_ok.setObjectName("PrimaryBtn")
        btn_ok.clicked.connect(self.accept)
        btn_cancel = QPushButton("ยกเลิก")
        btn_cancel.clicked.connect(self.reject)
        bottom_row.addWidget(btn_ok)
        bottom_row.addWidget(btn_cancel)
        layout.addLayout(bottom_row)

    def refresh_list(self, select_row=0):
        self.list_widget.clear()
        for cat in self.categories:
            color_label = cat["color"] if cat["color"] else "สีเดิม"
            item = QListWidgetItem(f"{cat['name']}   ({color_label})")
            if cat["color"]:
                item.setBackground(QBrush(QColor(cat["color"])))
                item.setForeground(QBrush(get_contrast_text_color(cat["color"])))
            self.list_widget.addItem(item)
        if 0 <= select_row < len(self.categories):
            self.list_widget.setCurrentRow(select_row)

    def on_select_row(self, row):
        if 0 <= row < len(self.categories):
            self.txt_name.setText(self.categories[row]["name"])

    def add_category(self):
        name = self.txt_name.text().strip()
        if not name:
            name = f"หมวดหมู่ใหม่ {len(self.categories) + 1}"
        self.next_id_num += 1
        new_cat = {"id": f"cat_{self.next_id_num}", "name": name, "color": "#E8F5E9"}
        self.categories.append(new_cat)
        self.refresh_list(len(self.categories) - 1)

    def update_category_name(self):
        row = self.list_widget.currentRow()
        name = self.txt_name.text().strip()
        if 0 <= row < len(self.categories) and name:
            self.categories[row]["name"] = name
            self.refresh_list(row)

    def pick_color(self):
        row = self.list_widget.currentRow()
        if not (0 <= row < len(self.categories)):
            return
        initial = QColor(self.categories[row]["color"]) if self.categories[row]["color"] else QColor("#FFF9C4")
        color = QColorDialog.getColor(initial, self, "เลือกสีไฮไลต์")
        if color.isValid():
            self.categories[row]["color"] = color.name().upper()
            self.refresh_list(row)

    def clear_color(self):
        row = self.list_widget.currentRow()
        if 0 <= row < len(self.categories):
            self.categories[row]["color"] = ""
            self.refresh_list(row)

    def delete_category(self):
        row = self.list_widget.currentRow()
        if len(self.categories) <= 1:
            QMessageBox.warning(self, "ลบไม่ได้", "ต้องมีชนิดเนื้อเรื่องอย่างน้อย 1 ชนิดครับ")
            return
        if 0 <= row < len(self.categories):
            self.categories.pop(row)
            self.refresh_list(max(0, row - 1))


# 1.5 หน้าต่างจัดการคลังคำศัพท์เฉพาะทางพระพุทธศาสนา
class GlossaryManagerDialog(QDialog):
    def __init__(self, words, parent=None):
        super().__init__(parent)
        self.setWindowTitle("คลังคำศัพท์เฉพาะทาง (พระพุทธศาสนา / ศัพท์เฉพาะ)")
        self.resize(540, 480)
        self.words = list(words)
        self.init_ui()
        self.refresh_list()

    def init_ui(self):
        layout = QVBoxLayout(self)

        lbl_info = QLabel(
            "คำศัพท์ในรายการนี้จะถูกส่งให้ AI (Whisper) ใช้เป็นแนวทางสะกดคำทุกครั้งที่ถอดเสียง\n"
            "สามารถพิมพ์เพิ่มทีละคำ หรือพิมพ์หลายคำคั่นด้วยเครื่องหมายจุลภาค (,) ได้เลย:"
        )
        lbl_info.setWordWrap(True)
        layout.addWidget(lbl_info)

        add_row = QHBoxLayout()
        self.txt_word = QLineEdit()
        self.txt_word.setPlaceholderText("เช่น ภุมมเทวา, อากาสเทวา, เคสสตั๊ดดี้, วิชชาธรรมกาย ...")
        self.txt_word.returnPressed.connect(self.add_words)
        btn_add = QPushButton("+ เพิ่มคำศัพท์")
        btn_add.setObjectName("PrimaryBtn")
        btn_add.clicked.connect(self.add_words)

        add_row.addWidget(self.txt_word, 1)
        add_row.addWidget(btn_add)
        layout.addLayout(add_row)

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        layout.addWidget(self.list_widget, 1)

        mid_row = QHBoxLayout()
        self.lbl_count = QLabel("ทั้งหมด: 0 คำ")
        btn_del = QPushButton("ลบคำที่เลือก")
        btn_del.clicked.connect(self.delete_selected)
        mid_row.addWidget(self.lbl_count)
        mid_row.addStretch()
        mid_row.addWidget(btn_del)
        layout.addLayout(mid_row)

        lbl_note = QLabel(
            "💡 คำแนะนำ: ควรเก็บคำเฉพาะที่มักถอดผิดประมาณ 20-50 คำสำคัญ เพื่อให้ Whisper โฟกัสได้แม่นยำที่สุด"
        )
        lbl_note.setStyleSheet("color: #64748B; font-size: 11px;")
        lbl_note.setWordWrap(True)
        layout.addWidget(lbl_note)

        bottom_row = QHBoxLayout()
        bottom_row.addStretch()
        btn_ok = QPushButton("บันทึกคลังคำศัพท์")
        btn_ok.setObjectName("PrimaryBtn")
        btn_ok.clicked.connect(self.accept)
        btn_cancel = QPushButton("ยกเลิก")
        btn_cancel.clicked.connect(self.reject)
        bottom_row.addWidget(btn_ok)
        bottom_row.addWidget(btn_cancel)
        layout.addLayout(bottom_row)

    def refresh_list(self):
        self.list_widget.clear()
        for w in self.words:
            self.list_widget.addItem(w)
        self.lbl_count.setText(f"ทั้งหมด: {len(self.words)} คำ")

    def add_words(self):
        raw = self.txt_word.text().strip()
        if not raw:
            return
        parts = [p.strip() for p in raw.replace("、", ",").split(",") if p.strip()]
        added = False
        for w in parts:
            if w not in self.words:
                self.words.append(w)
                added = True
        self.txt_word.clear()
        if added:
            self.refresh_list()
            self.list_widget.scrollToBottom()

    def delete_selected(self):
        selected_rows = sorted([idx.row() for idx in self.list_widget.selectedIndexes()], reverse=True)
        for r in selected_rows:
            if 0 <= r < len(self.words):
                self.words.pop(r)
        self.refresh_list()


# 2. Thread สำหรับถอดเสียงด้วย GPU
class TranscribeThread(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(list, float)
    error = pyqtSignal(str)

    def __init__(self, audio_path, default_cat_id, compute_type="int8_float16", batch_size=8, glossary_words=None):
        super().__init__()
        self.audio_path = audio_path
        self.default_cat_id = default_cat_id
        self.compute_type = compute_type
        self.batch_size = batch_size
        self.glossary_words = glossary_words or []

    def run(self):
        try:
            self.progress.emit(0, f"กำลังโหลดโมเดล Whisper (large-v3 / {self.compute_type}) ลง GPU...")
            model = WhisperModel("large-v3", device="cuda", compute_type=self.compute_type)

            initial_prompt = None
            if self.glossary_words:
                initial_prompt = "คำศัพท์เฉพาะทาง: " + ", ".join(self.glossary_words)

            use_batched = (self.batch_size > 1 and BatchedInferencePipeline is not None)
            if use_batched:
                self.progress.emit(
                    0, f"กำลังถอดเสียงด้วย Batched GPU (Batch {self.batch_size}, {self.compute_type})..."
                )
                batched_model = BatchedInferencePipeline(model=model)
                segments, info = batched_model.transcribe(
                    self.audio_path,
                    batch_size=self.batch_size,
                    chunk_length=8,
                    language="th",
                    initial_prompt=initial_prompt,
                    vad_parameters=dict(min_silence_duration_ms=250, speech_pad_ms=150)
                )
            else:
                mode_note = " (อัปเดต faster-whisper เพื่อเปิดโหมด Batched)" if (
                    self.batch_size > 1 and BatchedInferencePipeline is None
                ) else ""
                self.progress.emit(0, f"กำลังวิเคราะห์ไฟล์เสียงด้วย GPU...{mode_note}")
                segments, info = model.transcribe(
                    self.audio_path,
                    language="th",
                    initial_prompt=initial_prompt,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=250, max_speech_duration_s=8, speech_pad_ms=150)
                )

            total_duration = info.duration if info.duration and info.duration > 0 else 1.0
            results = []
            mode_tag = f"Batch {self.batch_size}" if use_batched else "Batch 1"
            for seg in segments:
                results.append({
                    "start": round(seg.start, 2),
                    "end": round(seg.end, 2),
                    "text": seg.text.strip(),
                    "cat_id": self.default_cat_id,
                    "checked": False
                })

                percent = min(int((seg.end / total_duration) * 100), 99)
                status_text = (
                    f"กำลังถอดเสียงด้วย GPU ({mode_tag})... {percent}% "
                    f"({format_hms(seg.end)} / {format_hms(total_duration)})"
                )
                self.progress.emit(percent, status_text)

            self.progress.emit(100, f"ถอดเสียงสำเร็จ! พบทั้งหมด {len(results)} ท่อน")
            self.finished.emit(results, total_duration)
        except Exception as e:
            self.error.emit(str(e))


# 3. Thread สำหรับตัดและรวมไฟล์เสียง
class AudioExportThread(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, ffmpeg_exe, input_audio, segments_to_cut, output_dir, base_name, fmt, mode,
                 pad_start=0.0, pad_end=0.0, max_duration=None):
        super().__init__()
        self.ffmpeg_exe = ffmpeg_exe
        self.input_audio = input_audio
        self.segments_to_cut = segments_to_cut
        self.output_dir = output_dir
        self.base_name = base_name
        self.fmt = fmt
        self.mode = mode
        self.pad_start = float(pad_start)
        self.pad_end = float(pad_end)
        self.max_duration = max_duration
        self._stop_flag = False
        self._current_proc = None

    def stop(self):
        self._stop_flag = True
        if self._current_proc and self._current_proc.poll() is None:
            try:
                self._current_proc.terminate()
            except Exception:
                pass

    def _codec_args(self):
        if self.fmt == "mp3":
            return ["-c:a", "libmp3lame", "-q:a", "2"]
        elif self.fmt == "m4a":
            return ["-c:a", "aac", "-b:a", "192k"]
        else:
            return ["-c:a", "pcm_s16le"]

    def _padded_bounds(self, seg):
        start = max(0.0, float(seg["start"]) - self.pad_start)
        end = float(seg["end"]) + self.pad_end
        if self.max_duration and self.max_duration > 0:
            end = min(end, self.max_duration)
        end = max(start + 0.05, end)
        return start, end

    def _run_cmd(self, cmd):
        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        self._current_proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, creationflags=creationflags
        )
        _, stderr = self._current_proc.communicate()
        return self._current_proc.returncode, stderr

    def run(self):
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            created_info = []

            if self.mode in ("merge", "both"):
                if self._stop_flag:
                    self.error.emit("ยกเลิกการทำงานแล้ว")
                    return
                self.progress.emit(f"กำลังตัดและรวม {len(self.segments_to_cut)} ท่อนเป็น 1 ไฟล์...")
                filter_parts = []
                concat_inputs = []
                for idx, seg in enumerate(self.segments_to_cut):
                    start, end = self._padded_bounds(seg)
                    filter_parts.append(f"[0:a]atrim=start={start:.3f}:end={end:.3f},asetpts=PTS-STARTPTS[a{idx}]")
                    concat_inputs.append(f"[a{idx}]")

                n = len(self.segments_to_cut)
                filter_complex = ";".join(filter_parts) + f";{''.join(concat_inputs)}concat=n={n}:v=0:a=1[outa]"
                out_file = os.path.join(self.output_dir, f"{self.base_name}_merged.{self.fmt}")
                cmd = [
                    self.ffmpeg_exe, "-y",
                    "-i", self.input_audio,
                    "-filter_complex", filter_complex,
                    "-map", "[outa]"
                ] + self._codec_args() + [out_file]

                code, stderr = self._run_cmd(cmd)
                if self._stop_flag:
                    self.error.emit("ยกเลิกการทำงานแล้ว")
                    return
                if code != 0:
                    self.error.emit(f"FFmpeg Error:\n{stderr[-500:]}")
                    return
                created_info.append(f"ไฟล์รวม: {out_file}")

            if self.mode in ("split", "both"):
                total = len(self.segments_to_cut)
                for idx, seg in enumerate(self.segments_to_cut, start=1):
                    if self._stop_flag:
                        self.error.emit("ยกเลิกการทำงานแล้ว")
                        return
                    self.progress.emit(f"กำลังตัดไฟล์ย่อยท่อนที่ {idx}/{total}...")
                    start, end = self._padded_bounds(seg)
                    out_file = os.path.join(self.output_dir, f"{self.base_name}_part{idx:03d}.{self.fmt}")
                    cmd = [
                        self.ffmpeg_exe, "-y",
                        "-i", self.input_audio,
                        "-ss", f"{start:.3f}",
                        "-to", f"{end:.3f}"
                    ] + self._codec_args() + [out_file]
                    code, stderr = self._run_cmd(cmd)
                    if code != 0:
                        self.error.emit(f"FFmpeg Error (ท่อนที่ {idx}):\n{stderr[-500:]}")
                        return
                created_info.append(f"ไฟล์ย่อยจำนวน {total} ไฟล์ในโฟลเดอร์ {self.output_dir}")

            self.finished.emit("\n".join(created_info))
        except Exception as e:
            self.error.emit(str(e))


# 4. หน้าต่างหลักของโปรแกรม (GPU Edition v13)
class SermonTranscriberApp(QMainWindow):
    COL_CHECK = 0
    COL_NO = 1
    COL_START = 2
    COL_END = 3
    COL_DUR = 4
    COL_CAT = 5
    COL_TEXT = 6

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sermon Timestamp Transcriber & Cutter (GPU Edition v13)")
        self.resize(1400, 830)

        self.audio_path = None
        self.audio_duration = 0.0
        self.transcript_data = []
        self.categories = load_categories()
        self.glossary_words = load_glossary()
        self.font_size = 13
        self.current_row = -1
        self._is_updating_table = False
        self._is_slider_dragging = False

        # สถานะสำหรับ Media Player เล่นท่อนที่เลือกต่อกัน
        self.play_queue = []
        self.play_queue_idx = -1
        self.play_mode_label = ""
        self.active_seg_start_ms = 0
        self.active_seg_end_ms = 0

        self.output_folder = os.path.join(os.path.expanduser("~"), "Downloads", "Sermon_Cut_Outputs")

        # ตัวเล่นเสียง
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(1.0)
        self.player.positionChanged.connect(self.on_player_position_changed)
        self.player.durationChanged.connect(self.on_player_duration_changed)

        self.apply_stylesheet()
        self.init_ui()
        self.apply_font_size()
        self.refresh_category_combos()
        self.update_bottom_status()

    def apply_stylesheet(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #F4F6F9;
                color: #1F2937;
                font-family: "Tahoma";
            }
            QGroupBox {
                background-color: #FFFFFF;
                border: 1px solid #D1D5DB;
                border-radius: 6px;
                margin-top: 12px;
                padding: 12px 10px 10px 10px;
                font-weight: bold;
                color: #1E3A5F;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #FFFFFF;
                border: 1px solid #CBD5E1;
                border-radius: 5px;
                padding: 5px 9px;
                color: #1F2937;
            }
            QPushButton:hover {
                background-color: #EFF6FF;
                border-color: #93C5FD;
            }
            QPushButton:disabled {
                background-color: #E5E7EB;
                color: #9CA3AF;
            }
            QPushButton#PrimaryBtn {
                background-color: #1E4E79;
                color: #FFFFFF;
                border: none;
                font-weight: bold;
            }
            QPushButton#PrimaryBtn:hover {
                background-color: #163B5C;
            }
            QPushButton#TealBtn {
                background-color: #0284C7;
                color: #FFFFFF;
                border: none;
                font-weight: bold;
            }
            QPushButton#TealBtn:hover {
                background-color: #0369A1;
            }
            QPushButton#PurpleBtn {
                background-color: #7E22CE;
                color: #FFFFFF;
                border: none;
                font-weight: bold;
            }
            QPushButton#PurpleBtn:hover {
                background-color: #6B21A8;
            }
            QPushButton#ExportBtn {
                background-color: #1B7A78;
                color: #FFFFFF;
                border: none;
                font-weight: bold;
                padding: 9px 12px;
            }
            QPushButton#ExportBtn:hover {
                background-color: #145E5C;
            }
            QPushButton#StopBtn {
                background-color: #374151;
                color: #FFFFFF;
                border: none;
                font-weight: bold;
                padding: 9px 12px;
            }
            QRadioButton {
                background-color: #F8FAFC;
                color: #334155;
                border: 1px solid #CBD5E1;
                border-radius: 5px;
                padding: 4px 8px;
                spacing: 6px;
                font-weight: normal;
            }
            QRadioButton:hover {
                background-color: #EFF6FF;
                border-color: #93C5FD;
            }
            QRadioButton:checked {
                background-color: #DBEAFE;
                color: #1E3A8A;
                border: 1px solid #2563EB;
                font-weight: bold;
            }
            QRadioButton::indicator {
                width: 14px;
                height: 14px;
                border-radius: 8px;
                border: 2px solid #64748B;
                background-color: #FFFFFF;
            }
            QRadioButton::indicator:checked {
                border: 4px solid #1E4E79;
                background-color: #FFFFFF;
            }
            QTableWidget::indicator {
                width: 16px;
                height: 16px;
                border-radius: 3px;
                border: 1px solid #64748B;
                background-color: #FFFFFF;
            }
            QTableWidget::indicator:checked {
                background-color: #1E4E79;
                border: 1px solid #1E4E79;
            }
            QTableWidget {
                background-color: #FFFFFF;
                border: 1px solid #D1D5DB;
                gridline-color: #E5E7EB;
                selection-background-color: #BAE6FD;
                selection-color: #0F172A;
            }
            QTableWidget::item:selected {
                background-color: #BAE6FD;
                color: #0F172A;
            }
            QHeaderView::section {
                background-color: #F8FAFC;
                border: none;
                border-bottom: 1px solid #CBD5E1;
                border-right: 1px solid #E5E7EB;
                padding: 6px;
                font-weight: bold;
                color: #334155;
            }
            QLineEdit, QComboBox {
                background-color: #FFFFFF;
                border: 1px solid #CBD5E1;
                border-radius: 4px;
                padding: 4px 6px;
            }
            QLineEdit#StepInput {
                background-color: #FFFFFF;
                color: #0F172A;
                border: 1px solid #94A3B8;
                border-radius: 4px;
                padding: 2px 4px;
                font-size: 14px;
                font-weight: bold;
                min-height: 24px;
            }
            QPushButton#StepMiniBtn {
                background-color: #F1F5F9;
                color: #0F172A;
                border: 1px solid #94A3B8;
                border-radius: 3px;
                padding: 0px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton#StepMiniBtn:hover {
                background-color: #DBEAFE;
                border-color: #2563EB;
            }
            QProgressBar {
                border: 1px solid #CBD5E1;
                border-radius: 4px;
                background-color: #E5E7EB;
                text-align: center;
                height: 18px;
            }
            QProgressBar::chunk {
                background-color: #1B7A78;
                border-radius: 3px;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #E2E8F0;
                border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: #1E4E79;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #1E4E79;
                width: 14px;
                height: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
            QFrame#BottomBar {
                background-color: #FFFFFF;
                border-top: 1px solid #D1D5DB;
            }
        """)

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_vbox = QVBoxLayout(central)
        main_vbox.setContentsMargins(10, 8, 10, 6)
        main_vbox.setSpacing(6)

        # ================= แถบหัวโปรแกรมด้านบนสุด =================
        header_bar = QHBoxLayout()
        title_lbl = QLabel("Sermon Timestamp Transcriber & Cutter")
        title_lbl.setStyleSheet("font-size: 17px; font-weight: bold; color: #0F172A;")
        header_bar.addWidget(title_lbl)

        header_bar.addSpacing(14)
        self.txt_search = QLineEdit()
        self.txt_search.setPlaceholderText("🔍 ค้นหาคำในบทเทศน์...")
        self.txt_search.setFixedWidth(250)
        self.txt_search.textChanged.connect(self.filter_table_by_search)
        header_bar.addWidget(self.txt_search)

        btn_clear_search = QPushButton("✕")
        btn_clear_search.setFixedWidth(28)
        btn_clear_search.clicked.connect(lambda: self.txt_search.clear())
        header_bar.addWidget(btn_clear_search)

        header_bar.addStretch()

        self.lbl_font = QLabel(f"ฟอนต์: ({self.font_size} pt)")
        self.btn_font_minus = QPushButton("ก- (ลด)")
        self.btn_font_minus.clicked.connect(self.decrease_font)
        self.btn_font_plus = QPushButton("ก+ (ขยาย)")
        self.btn_font_plus.clicked.connect(self.increase_font)

        header_bar.addWidget(self.lbl_font)
        header_bar.addWidget(self.btn_font_minus)
        header_bar.addWidget(self.btn_font_plus)
        main_vbox.addLayout(header_bar)

        # ================= พื้นที่ตรงกลาง: ซ้าย (ตารางบทถอดความ) | ขวา (แถบจัดการ) =================
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ---------- ฝั่งซ้าย: แถบควบคุมด้านบน + ตารางบทถอดความ ----------
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 4, 0)
        left_layout.setSpacing(5)

        # แถบเครื่องมือด้านบนตาราง
        top_tool_1 = QHBoxLayout()
        top_tool_1.setSpacing(5)
        btn_all = QPushButton("[✓] ทั้งหมด")
        btn_all.clicked.connect(lambda: self.set_all_checks(True))
        btn_none = QPushButton("[ ] ยกเลิก")
        btn_none.clicked.connect(lambda: self.set_all_checks(False))
        btn_invert = QPushButton("[⇄] สลับเลือก")
        btn_invert.clicked.connect(self.invert_checks)

        btn_merge_rows = QPushButton("รวมท่อนที่ติ๊ก (Merge)")
        btn_merge_rows.setObjectName("PrimaryBtn")
        btn_merge_rows.clicked.connect(self.merge_checked_rows)

        self.btn_add_segment = QPushButton("+ เพิ่มท่อนเนื้อเรื่อง")
        self.btn_add_segment.setToolTip("เพิ่มท่อนเนื้อเรื่องใหม่ด้วยตัวเอง (เช่น แทรกหรือเพิ่มช่วงเวลาที่ต้องการปรับเสียงเอง)")
        self.btn_add_segment.clicked.connect(self.add_custom_segment)

        self.btn_del_segment = QPushButton("- ลบท่อนเนื้อเรื่อง")
        self.btn_del_segment.setToolTip("ลบท่อนเนื้อเรื่องที่ไฮไลต์เลือกอยู่ (หรือท่อนที่ติ๊กเลือกไว้)")
        self.btn_del_segment.clicked.connect(self.delete_selected_segments)

        top_tool_1.addWidget(btn_all)
        top_tool_1.addWidget(btn_none)
        top_tool_1.addWidget(btn_invert)
        top_tool_1.addWidget(btn_merge_rows)
        top_tool_1.addWidget(self.btn_add_segment)
        top_tool_1.addWidget(self.btn_del_segment)

        top_tool_1.addSpacing(6)
        top_tool_1.addWidget(QLabel("ชนิด:"))
        self.combo_batch_cat = QComboBox()
        self.combo_batch_cat.setMinimumWidth(110)
        self.combo_batch_cat.setToolTip("ไฮไลต์แถวที่ต้องการ (แถวเดียวหรือหลายแถวด้วย Shift/Ctrl) แล้วเลือกชนิดที่นี่เพื่อเปลี่ยนทันที")
        self.combo_batch_cat.activated.connect(self.on_combo_category_activated)
        self.btn_manage_cat = QPushButton("🎨 ชนิด/สี")
        self.btn_manage_cat.setToolTip("จัดการเพิ่ม/แก้ไขชื่อและสีของชนิดเนื้อเรื่อง")
        self.btn_manage_cat.clicked.connect(self.open_category_manager)

        btn_check_cat = QPushButton("ติ๊กเฉพาะชนิดนี้")
        btn_check_cat.clicked.connect(self.check_rows_by_category)

        top_tool_1.addWidget(self.combo_batch_cat)
        top_tool_1.addWidget(self.btn_manage_cat)
        top_tool_1.addWidget(btn_check_cat)

        # ช่องปรับเวลา เริ่ม / จบ (ใส่ค่าก้าวกระโดดฟิกซ์ไว้เอง เช่น 1, 2, 3 แล้วกด + หรือ - เพื่อปรับเวลาท่อนที่เลือกทันที)
        top_tool_1.addSpacing(6)
        lbl_pad_s = QLabel("เริ่ม:")
        lbl_pad_s.setStyleSheet("color: #1E3A5F; font-size: 14px; font-weight: bold;")
        top_tool_1.addWidget(lbl_pad_s)

        self.txt_step_start = QLineEdit("1")
        self.txt_step_start.setObjectName("StepInput")
        self.txt_step_start.setFixedWidth(44)
        self.txt_step_start.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.txt_step_start.setToolTip("พิมพ์จำนวนวินาทีที่ต้องการเพิ่ม/ลดฟิกซ์ไว้ (เช่น 1, 2, 3) แล้วกดปุ่ม + หรือ - ด้านข้าง")
        top_tool_1.addWidget(self.txt_step_start)

        s_btn_col = QVBoxLayout()
        s_btn_col.setSpacing(1)
        s_btn_col.setContentsMargins(0, 0, 0, 0)
        btn_s_plus = QPushButton("+")
        btn_s_plus.setObjectName("StepMiniBtn")
        btn_s_plus.setFixedSize(22, 14)
        btn_s_plus.setToolTip("เพิ่มเวลาเริ่มของท่อนที่เลือกอยู่ตามจำนวนวินาทีในช่อง")
        btn_s_plus.clicked.connect(lambda: self.adjust_selected_time("start", 1))
        btn_s_minus = QPushButton("-")
        btn_s_minus.setObjectName("StepMiniBtn")
        btn_s_minus.setFixedSize(22, 14)
        btn_s_minus.setToolTip("ลดเวลาเริ่มของท่อนที่เลือกอยู่ตามจำนวนวินาทีในช่อง")
        btn_s_minus.clicked.connect(lambda: self.adjust_selected_time("start", -1))
        s_btn_col.addWidget(btn_s_plus)
        s_btn_col.addWidget(btn_s_minus)
        top_tool_1.addLayout(s_btn_col)

        lbl_unit_s = QLabel("วิ")
        lbl_unit_s.setStyleSheet("color: #0F172A; font-size: 14px;")
        top_tool_1.addWidget(lbl_unit_s)

        lbl_pad_e = QLabel("จบ:")
        lbl_pad_e.setStyleSheet("color: #1E3A5F; font-size: 14px; font-weight: bold;")
        top_tool_1.addWidget(lbl_pad_e)

        self.txt_step_end = QLineEdit("1")
        self.txt_step_end.setObjectName("StepInput")
        self.txt_step_end.setFixedWidth(44)
        self.txt_step_end.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.txt_step_end.setToolTip("พิมพ์จำนวนวินาทีที่ต้องการเพิ่ม/ลดฟิกซ์ไว้ (เช่น 1, 2, 3) แล้วกดปุ่ม + หรือ - ด้านข้าง")
        top_tool_1.addWidget(self.txt_step_end)

        e_btn_col = QVBoxLayout()
        e_btn_col.setSpacing(1)
        e_btn_col.setContentsMargins(0, 0, 0, 0)
        btn_e_plus = QPushButton("+")
        btn_e_plus.setObjectName("StepMiniBtn")
        btn_e_plus.setFixedSize(22, 14)
        btn_e_plus.setToolTip("เพิ่มเวลาจบของท่อนที่เลือกอยู่ตามจำนวนวินาทีในช่อง")
        btn_e_plus.clicked.connect(lambda: self.adjust_selected_time("end", 1))
        btn_e_minus = QPushButton("-")
        btn_e_minus.setObjectName("StepMiniBtn")
        btn_e_minus.setFixedSize(22, 14)
        btn_e_minus.setToolTip("ลดเวลาจบของท่อนที่เลือกอยู่ตามจำนวนวินาทีในช่อง")
        btn_e_minus.clicked.connect(lambda: self.adjust_selected_time("end", -1))
        e_btn_col.addWidget(btn_e_plus)
        e_btn_col.addWidget(btn_e_minus)
        top_tool_1.addLayout(e_btn_col)

        lbl_unit_e = QLabel("วิ")
        lbl_unit_e.setStyleSheet("color: #0F172A; font-size: 14px;")
        top_tool_1.addWidget(lbl_unit_e)

        top_tool_1.addStretch()
        left_layout.addLayout(top_tool_1)

        # ตารางบทถอดคำพูด
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels([
            "สถานะ", "ลำดับ", "เวลาเริ่ม", "เวลาจบ", "ความยาว", "ชนิด", "ข้อความคำพูด / รายละเอียดคำเทศน์"
        ])
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(self.COL_CHECK, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(self.COL_NO, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(self.COL_START, QHeaderView.ResizeMode.Interactive)
        hdr.setSectionResizeMode(self.COL_END, QHeaderView.ResizeMode.Interactive)
        hdr.setSectionResizeMode(self.COL_DUR, QHeaderView.ResizeMode.Interactive)
        hdr.setSectionResizeMode(self.COL_CAT, QHeaderView.ResizeMode.Interactive)
        hdr.setSectionResizeMode(self.COL_TEXT, QHeaderView.ResizeMode.Stretch)
        self.table.setColumnWidth(self.COL_START, 115)
        self.table.setColumnWidth(self.COL_END, 115)
        self.table.setColumnWidth(self.COL_DUR, 95)
        self.table.setColumnWidth(self.COL_CAT, 105)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.EditKeyPressed)
        self.table.setWordWrap(True)
        self.table.verticalHeader().setVisible(False)
        self.table.cellClicked.connect(self.on_table_cell_clicked)
        self.table.cellDoubleClicked.connect(self.on_table_double_clicked)
        self.table.itemChanged.connect(self.on_item_changed)
        self.table.installEventFilter(self)
        self.table.viewport().installEventFilter(self)
        left_layout.addWidget(self.table)

        splitter.addWidget(left_panel)

        # ---------- ฝั่งขวา: แถบจัดการทั้งหมด ----------
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QFrame.Shape.NoFrame)
        right_scroll.setMinimumWidth(410)
        right_scroll.setMaximumWidth(510)

        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(4, 0, 2, 0)
        right_layout.setSpacing(8)

        # --- กล่อง 1: แหล่งข้อมูล & ถอดเสียง (GPU) ---
        grp_source = QGroupBox("1. แหล่งข้อมูล (ไฟล์เสียงในเครื่อง & GPU)")
        g1_layout = QVBoxLayout(grp_source)

        btn_row1 = QHBoxLayout()
        self.btn_open = QPushButton("เปิดไฟล์เสียงในเครื่อง")
        self.btn_open.setObjectName("PrimaryBtn")
        self.btn_open.clicked.connect(self.open_file)

        self.btn_transcribe = QPushButton("เริ่มถอดข้อความ (GPU)")
        self.btn_transcribe.setObjectName("PurpleBtn")
        self.btn_transcribe.setEnabled(False)
        self.btn_transcribe.clicked.connect(self.start_transcription)

        btn_row1.addWidget(self.btn_open, 1)
        btn_row1.addWidget(self.btn_transcribe, 1)
        g1_layout.addLayout(btn_row1)

        gpu_row = QHBoxLayout()
        gpu_row.addWidget(QLabel("โปรไฟล์ GPU:"))
        self.combo_gpu_profile = QComboBox()
        for prof in GPU_PROFILES:
            self.combo_gpu_profile.addItem(prof["label"], prof)
        self.combo_gpu_profile.setCurrentIndex(0)
        gpu_row.addWidget(self.combo_gpu_profile, 1)
        g1_layout.addLayout(gpu_row)

        self.btn_glossary = QPushButton(f"📖 คลังคำศัพท์เฉพาะทางพระพุทธศาสนา ({len(self.glossary_words)} คำ)")
        self.btn_glossary.clicked.connect(self.open_glossary_manager)
        g1_layout.addWidget(self.btn_glossary)

        proj_row = QHBoxLayout()
        self.btn_import_proj = QPushButton("📂 Import ที่เคยถอดไว้")
        self.btn_import_proj.clicked.connect(self.import_project)

        self.btn_save_proj = QPushButton("💾 บันทึกโปรเจกต์")
        self.btn_save_proj.setEnabled(False)
        self.btn_save_proj.clicked.connect(self.save_project)

        proj_row.addWidget(self.btn_import_proj, 1)
        proj_row.addWidget(self.btn_save_proj, 1)
        g1_layout.addLayout(proj_row)

        self.lbl_file_info = QLabel("ยังไม่ได้เลือกไฟล์เสียง (.mp3 / .m4a / .wav)")
        self.lbl_file_info.setWordWrap(True)
        self.lbl_file_info.setStyleSheet("color: #334155; font-weight: normal; padding: 2px 0;")
        g1_layout.addWidget(self.lbl_file_info)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        g1_layout.addWidget(self.progress_bar)

        self.lbl_progress_status = QLabel("พร้อมใช้งาน")
        self.lbl_progress_status.setWordWrap(True)
        self.lbl_progress_status.setStyleSheet("color: #475569; font-weight: normal; font-size: 12px;")
        g1_layout.addWidget(self.lbl_progress_status)

        right_layout.addWidget(grp_source)

        # --- กล่อง 2: มีเดียเพลย์เยอร์ (พรีวิวฟังท่อนที่เลือกต่อกันก่อน Export) ---
        grp_player = QGroupBox("2. มีเดียเพลย์เยอร์ (พรีวิวฟังท่อนที่เลือกต่อกัน)")
        g2_layout = QVBoxLayout(grp_player)

        self.lbl_player_now = QLabel("สถานะ: ยังไม่ได้เล่น (ติ๊กเลือกท่อนแล้วกดฟังต่อกันได้เลย)")
        self.lbl_player_now.setWordWrap(True)
        self.lbl_player_now.setStyleSheet("color: #1E3A5F; font-weight: bold; font-size: 12px;")
        g2_layout.addWidget(self.lbl_player_now)

        slider_row = QHBoxLayout()
        self.lbl_time_cur = QLabel("00:00:00")
        self.lbl_time_cur.setStyleSheet("font-weight: normal; font-size: 11px;")
        self.slider_pos = QSlider(Qt.Orientation.Horizontal)
        self.slider_pos.setRange(0, 1000)
        self.slider_pos.sliderPressed.connect(self.on_slider_pressed)
        self.slider_pos.sliderReleased.connect(self.on_slider_released)
        self.lbl_time_tot = QLabel("00:00:00")
        self.lbl_time_tot.setStyleSheet("font-weight: normal; font-size: 11px;")

        slider_row.addWidget(self.lbl_time_cur)
        slider_row.addWidget(self.slider_pos, 1)
        slider_row.addWidget(self.lbl_time_tot)
        g2_layout.addLayout(slider_row)

        play_mode_row = QHBoxLayout()
        self.btn_play_queue = QPushButton("▶ ฟังที่ติ๊กทั้งหมด")
        self.btn_play_queue.setObjectName("PrimaryBtn")
        self.btn_play_queue.setToolTip("พรีวิวเล่นท่อนที่ติ๊กถูกไว้ทั้งหมดเรียงต่อกันตั้งแต่ต้นจนจบ")
        self.btn_play_queue.setEnabled(False)
        self.btn_play_queue.clicked.connect(self.play_checked_sequentially)

        self.btn_play_selected = QPushButton("▶ ฟังเฉพาะที่เลือกอยู่")
        self.btn_play_selected.setObjectName("TealBtn")
        self.btn_play_selected.setToolTip("พรีวิวเล่นต่อกันเฉพาะท่อนที่ไฮไลต์แถบสีฟ้าอยู่ในตารางขณะนี้ (คลิก / Shift / Ctrl)")
        self.btn_play_selected.setEnabled(False)
        self.btn_play_selected.clicked.connect(self.play_selected_rows_sequentially)

        play_mode_row.addWidget(self.btn_play_queue, 1)
        play_mode_row.addWidget(self.btn_play_selected, 1)
        g2_layout.addLayout(play_mode_row)

        ctrl_row = QHBoxLayout()
        self.btn_prev_seg = QPushButton("⏮ ก่อนหน้า")
        self.btn_prev_seg.setEnabled(False)
        self.btn_prev_seg.clicked.connect(self.play_prev_in_queue)

        self.btn_pause_resume = QPushButton("⏸ พัก / ต่อ")
        self.btn_pause_resume.setEnabled(False)
        self.btn_pause_resume.clicked.connect(self.toggle_pause_resume)

        self.btn_stop_player = QPushButton("⏹ หยุด")
        self.btn_stop_player.setEnabled(False)
        self.btn_stop_player.clicked.connect(self.stop_player)

        self.btn_next_seg = QPushButton("ถัดไป ⏭")
        self.btn_next_seg.setEnabled(False)
        self.btn_next_seg.clicked.connect(self.play_next_in_queue)

        ctrl_row.addWidget(self.btn_prev_seg)
        ctrl_row.addWidget(self.btn_pause_resume)
        ctrl_row.addWidget(self.btn_stop_player)
        ctrl_row.addWidget(self.btn_next_seg)
        g2_layout.addLayout(ctrl_row)

        right_layout.addWidget(grp_player)

        # --- กล่อง 3: รูปแบบไฟล์ & Export ---
        grp_export = QGroupBox("3. รูปแบบไฟล์ & เริ่มตัดรวม (Export)")
        g3_layout = QVBoxLayout(grp_export)

        fmt_row = QHBoxLayout()
        fmt_row.addWidget(QLabel("ไฟล์:"))
        self.fmt_group = QButtonGroup(self)
        self.rb_mp3 = QRadioButton(".mp3")
        self.rb_m4a = QRadioButton(".m4a")
        self.rb_wav = QRadioButton(".wav")
        self.rb_mp3.setChecked(True)
        for rb in (self.rb_mp3, self.rb_m4a, self.rb_wav):
            self.fmt_group.addButton(rb)
            fmt_row.addWidget(rb)
        fmt_row.addStretch()
        g3_layout.addLayout(fmt_row)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("โหมด:"))
        self.mode_group = QButtonGroup(self)
        self.rb_merge = QRadioButton("รวม 1 ไฟล์")
        self.rb_split = QRadioButton("แยกไฟล์ย่อย")
        self.rb_both = QRadioButton("ทั้งคู่")
        self.rb_merge.setChecked(True)
        for rb in (self.rb_merge, self.rb_split, self.rb_both):
            self.mode_group.addButton(rb)
            mode_row.addWidget(rb)
        mode_row.addStretch()
        g3_layout.addLayout(mode_row)

        folder_row = QHBoxLayout()
        folder_row.addWidget(QLabel("โฟลเดอร์:"))
        self.txt_output_folder = QLineEdit(self.output_folder)
        btn_browse_folder = QPushButton("เลือก...")
        btn_browse_folder.clicked.connect(self.choose_output_folder)
        folder_row.addWidget(self.txt_output_folder, 1)
        folder_row.addWidget(btn_browse_folder)
        g3_layout.addLayout(folder_row)

        exp_btn_row = QHBoxLayout()
        self.btn_export_audio = QPushButton("⬇ Export ไฟล์เสียงที่ติ๊ก")
        self.btn_export_audio.setObjectName("ExportBtn")
        self.btn_export_audio.setEnabled(False)
        self.btn_export_audio.clicked.connect(self.export_selected_audio)

        self.btn_stop_export = QPushButton("Stop")
        self.btn_stop_export.setObjectName("StopBtn")
        self.btn_stop_export.setEnabled(False)
        self.btn_stop_export.clicked.connect(self.stop_audio_export)

        exp_btn_row.addWidget(self.btn_export_audio, 3)
        exp_btn_row.addWidget(self.btn_stop_export, 1)
        g3_layout.addLayout(exp_btn_row)

        self.btn_export_text = QPushButton("📄 Export บทถอดความ (HTML สี / Excel CSV / TXT / SRT)")
        self.btn_export_text.setEnabled(False)
        self.btn_export_text.clicked.connect(self.export_transcript)
        g3_layout.addWidget(self.btn_export_text)

        right_layout.addWidget(grp_export)
        right_layout.addStretch()

        right_scroll.setWidget(right_container)
        splitter.addWidget(right_scroll)

        splitter.setStretchFactor(0, 7)
        splitter.setStretchFactor(1, 3)
        main_vbox.addWidget(splitter, 1)

        # ================= แถบสถานะด้านล่างสุด =================
        bottom_frame = QFrame()
        bottom_frame.setObjectName("BottomBar")
        bottom_layout = QHBoxLayout(bottom_frame)
        bottom_layout.setContentsMargins(8, 5, 8, 5)

        self.lbl_bottom_summary = QLabel("เลือกแล้ว: 0 / 0 ท่อน (00:00:00)")
        self.lbl_bottom_summary.setStyleSheet("font-weight: bold; color: #0F172A; font-size: 13px;")

        lbl_tip = QLabel("💡 Shift/Ctrl เลือกหลายท่อน | Spacebar ติ๊กเลือก/ยกเลิก | ดับเบิลคลิกหรือคลิกขวาเพื่อติ๊กเพิ่ม | F2 แก้เวลา/ข้อความ")
        lbl_tip.setStyleSheet("color: #64748B; font-size: 12px;")

        bottom_layout.addWidget(self.lbl_bottom_summary)
        bottom_layout.addStretch()
        bottom_layout.addWidget(lbl_tip)
        main_vbox.addWidget(bottom_frame)

    # ---------- ฟังก์ชันเปลี่ยนชนิด + ติ๊ก/ยกเลิกติ๊ก ----------
    def apply_category_to_rows(self, rows, cat_id, update_check=None):
        """เปลี่ยนชนิดและสีของแถวที่ระบุทันที (ใช้ได้ทั้งแถวเดียวและหลายแถวที่ไฮไลต์อยู่)"""
        if not rows or not cat_id:
            return
        cat = self.get_category_by_id(cat_id)
        self._is_updating_table = True
        changed_count = 0
        for r in rows:
            if 0 <= r < len(self.transcript_data):
                self.transcript_data[r]["cat_id"] = cat["id"]
                if update_check is not None:
                    self.transcript_data[r]["checked"] = bool(update_check)
                    chk_cell = self.table.item(r, self.COL_CHECK)
                    if chk_cell:
                        chk_cell.setCheckState(
                            Qt.CheckState.Checked if update_check else Qt.CheckState.Unchecked
                        )
                cat_cell = self.table.item(r, self.COL_CAT)
                if cat_cell:
                    cat_cell.setText(cat["name"])
                self.apply_row_color(r, cat["color"])
                changed_count += 1
        self._is_updating_table = False
        self.update_bottom_status()
        if changed_count > 0:
            self.lbl_progress_status.setText(
                f"เปลี่ยน {changed_count} ท่อนที่ไฮไลต์เป็นชนิด '{cat['name']}' แล้ว"
            )

    def on_combo_category_activated(self, index):
        """เมื่อเลือกชนิดในช่อง ชนิด: ให้เปลี่ยนแถวที่กำลังไฮไลต์อยู่ทันที"""
        if index < 0:
            return
        target_cat_id = self.combo_batch_cat.itemData(index)
        if not target_cat_id:
            return
        highlighted_rows = sorted(set(
            idx.row() for idx in self.table.selectedIndexes()
            if 0 <= idx.row() < len(self.transcript_data)
        ))
        if not highlighted_rows and 0 <= self.current_row < len(self.transcript_data):
            highlighted_rows = [self.current_row]
        if highlighted_rows:
            self.apply_category_to_rows(highlighted_rows, target_cat_id)

    def apply_check_and_color(self, rows, check_state, cat_id=None):
        """ตั้งค่า checked โดยคงชนิดและสีเดิมไว้เสมอ"""
        if not rows:
            return
        self._is_updating_table = True
        last_cat_name = ""
        for r in rows:
            if 0 <= r < len(self.transcript_data):
                self.transcript_data[r]["checked"] = check_state
                chk_cell = self.table.item(r, self.COL_CHECK)
                if chk_cell:
                    chk_cell.setCheckState(Qt.CheckState.Checked if check_state else Qt.CheckState.Unchecked)

                if check_state and cat_id:
                    cat = self.get_category_by_id(cat_id)
                    self.transcript_data[r]["cat_id"] = cat["id"]
                else:
                    cat = self.get_category_by_id(
                        self.transcript_data[r].get("cat_id", self.categories[0]["id"])
                    )
                cat_cell = self.table.item(r, self.COL_CAT)
                if cat_cell:
                    cat_cell.setText(cat["name"])
                self.apply_row_color(r, cat["color"])
                last_cat_name = cat["name"]
        self._is_updating_table = False
        self.update_bottom_status()

        if check_state:
            self.lbl_progress_status.setText(f"ติ๊กเลือก {len(rows)} ท่อนแล้ว")
        else:
            self.lbl_progress_status.setText(f"ยกเลิกการเลือก {len(rows)} ท่อนแล้ว (คงชนิดและสีเดิมไว้)")

    def eventFilter(self, obj, event):
        if obj is self.table and event.type() == QEvent.Type.KeyPress:
            if event.key() == Qt.Key.Key_Space:
                selected_rows = sorted(set(idx.row() for idx in self.table.selectedIndexes()))
                if selected_rows:
                    all_already_checked = all(
                        self.transcript_data[r].get("checked", False)
                        for r in selected_rows if 0 <= r < len(self.transcript_data)
                    )
                    if all_already_checked:
                        self.apply_check_and_color(selected_rows, False)
                    else:
                        self.apply_check_and_color(selected_rows, True)
                return True

        if obj is self.table.viewport() and event.type() == QEvent.Type.MouseButtonPress:
            if event.button() == Qt.MouseButton.RightButton:
                pos = event.position().toPoint()
                row = self.table.rowAt(pos.y())
                if 0 <= row < len(self.transcript_data):
                    selected_rows = sorted(set(idx.row() for idx in self.table.selectedIndexes()))
                    if row in selected_rows and len(selected_rows) > 1:
                        target_rows = selected_rows
                    else:
                        target_rows = [row]
                        self.table.selectRow(row)
                    self.current_row = row
                    self.apply_check_and_color(target_rows, True)
                return True

        return super().eventFilter(obj, event)

    def get_category_by_id(self, cat_id):
        for c in self.categories:
            if c["id"] == cat_id:
                return c
        return self.categories[0]

    def refresh_category_combos(self):
        current_data = self.combo_batch_cat.currentData()
        self.combo_batch_cat.clear()
        for c in self.categories:
            self.combo_batch_cat.addItem(c["name"], c["id"])
        if current_data:
            idx = self.combo_batch_cat.findData(current_data)
            if idx >= 0:
                self.combo_batch_cat.setCurrentIndex(idx)

    def open_category_manager(self):
        dlg = CategoryManagerDialog(self.categories, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.categories = dlg.categories
            save_categories(self.categories)
            self.refresh_category_combos()
            self.rebuild_table_from_data()

    def open_glossary_manager(self):
        dlg = GlossaryManagerDialog(self.glossary_words, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.glossary_words = dlg.words
            save_glossary(self.glossary_words)
            self.btn_glossary.setText(f"📖 คลังคำศัพท์เฉพาะทางพระพุทธศาสนา ({len(self.glossary_words)} คำ)")
            self.lbl_progress_status.setText(f"บันทึกคลังคำศัพท์แล้ว ({len(self.glossary_words)} คำ)")

    def adjust_time_columns_width(self):
        fm = self.table.fontMetrics()
        time_w = fm.horizontalAdvance("00:00:00.0") + 28
        dur_w = fm.horizontalAdvance("00:00:00") + 24
        cat_w = max(95, fm.horizontalAdvance("เนื้อเรื่อง") + 32)
        self.table.setColumnWidth(self.COL_START, max(105, time_w))
        self.table.setColumnWidth(self.COL_END, max(105, time_w))
        self.table.setColumnWidth(self.COL_DUR, max(85, dur_w))
        self.table.setColumnWidth(self.COL_CAT, cat_w)

    def apply_font_size(self):
        font = QFont("Tahoma", self.font_size)
        self.table.setFont(font)
        self.adjust_time_columns_width()
        self.table.resizeRowsToContents()
        self.lbl_font.setText(f"ฟอนต์: ({self.font_size} pt)")

    def increase_font(self):
        if self.font_size < 34:
            self.font_size += 2
            self.apply_font_size()

    def decrease_font(self):
        if self.font_size > 9:
            self.font_size -= 2
            self.apply_font_size()

    def set_player_controls_enabled(self, enabled):
        self.btn_play_queue.setEnabled(enabled)
        self.btn_play_selected.setEnabled(enabled)
        self.btn_prev_seg.setEnabled(enabled)
        self.btn_pause_resume.setEnabled(enabled)
        self.btn_stop_player.setEnabled(enabled)
        self.btn_next_seg.setEnabled(enabled)

    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "เลือกไฟล์เสียง", "", "Audio Files (*.mp3 *.m4a *.wav)"
        )
        if file_path:
            self.stop_player()
            self.audio_path = file_path
            self.player.setSource(QUrl.fromLocalFile(file_path))
            fname = os.path.basename(file_path)
            self.lbl_file_info.setText(f"ไฟล์: {fname}")
            self.lbl_progress_status.setText("พร้อมถอดเสียงด้วย GPU")
            self.progress_bar.setValue(0)
            self.btn_transcribe.setEnabled(True)
            self.set_player_controls_enabled(True)

            auto_proj = os.path.splitext(file_path)[0] + ".sermon.json"
            if os.path.exists(auto_proj):
                reply = QMessageBox.question(
                    self,
                    "พบไฟล์ที่เคยถอดความไว้แล้ว",
                    f"พบไฟล์ที่เคยถอดความไว้สำหรับเสียงนี้:\n{os.path.basename(auto_proj)}\n\nต้องการโหลดเข้ามาเลยโดยไม่ต้องถอดเสียงใหม่ไหมครับ?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.Yes:
                    self.load_project_from_file(auto_proj)

    def start_transcription(self):
        if not self.audio_path:
            return
        self.stop_player()
        prof = self.combo_gpu_profile.currentData() or GPU_PROFILES[0]
        self.progress_bar.setValue(0)
        self.lbl_progress_status.setText(
            f"กำลังเตรียมถอดเสียงด้วย GPU ({prof['compute_type']}, Batch {prof['batch_size']})..."
        )
        self.btn_open.setEnabled(False)
        self.btn_transcribe.setEnabled(False)
        self.combo_gpu_profile.setEnabled(False)
        self.btn_glossary.setEnabled(False)
        self.btn_import_proj.setEnabled(False)
        self.btn_save_proj.setEnabled(False)
        self.btn_export_text.setEnabled(False)
        self.btn_export_audio.setEnabled(False)

        default_cat_id = self.categories[0]["id"]
        self.thread = TranscribeThread(
            self.audio_path,
            default_cat_id,
            compute_type=prof["compute_type"],
            batch_size=prof["batch_size"],
            glossary_words=self.glossary_words,
        )
        self.thread.progress.connect(self.update_progress)
        self.thread.finished.connect(self.on_transcription_done)
        self.thread.error.connect(self.on_transcription_error)
        self.thread.start()

    def update_progress(self, percent, status_text):
        self.progress_bar.setValue(percent)
        self.lbl_progress_status.setText(status_text)

    def on_transcription_done(self, results, total_duration):
        self.transcript_data = results
        self.audio_duration = total_duration
        fname = os.path.basename(self.audio_path) if self.audio_path else ""
        self.lbl_file_info.setText(
            f"ไฟล์: {fname}\nความยาวรวม: {format_hms(total_duration)} | ทั้งหมด {len(results)} ท่อน"
        )
        self.rebuild_table_from_data()
        self.progress_bar.setValue(100)
        self.btn_open.setEnabled(True)
        self.btn_transcribe.setEnabled(True)
        self.combo_gpu_profile.setEnabled(True)
        self.btn_glossary.setEnabled(True)
        self.btn_import_proj.setEnabled(True)
        self.btn_save_proj.setEnabled(True)
        self.btn_export_text.setEnabled(True)
        self.btn_export_audio.setEnabled(True)
        self.set_player_controls_enabled(True)

        if self.audio_path:
            auto_proj = os.path.splitext(self.audio_path)[0] + ".sermon.json"
            self.write_project_file(auto_proj, silent=True)
            self.lbl_progress_status.setText(
                f"ถอดเสียงสำเร็จ ({len(results)} ท่อน) และบันทึกไฟล์โปรเจกต์อัตโนมัติแล้ว"
            )

    def on_transcription_error(self, err_msg):
        QMessageBox.critical(self, "เกิดข้อผิดพลาด", err_msg)
        self.lbl_progress_status.setText("เกิดข้อผิดพลาดในการถอดเสียง")
        self.progress_bar.setValue(0)
        self.btn_open.setEnabled(True)
        self.btn_transcribe.setEnabled(True)
        self.combo_gpu_profile.setEnabled(True)
        self.btn_glossary.setEnabled(True)
        self.btn_import_proj.setEnabled(True)

    # ---------- ระบบ Import / Save โปรเจกต์ ----------
    def write_project_file(self, file_path, silent=False):
        try:
            payload = {
                "audio_path": self.audio_path,
                "audio_duration": self.audio_duration,
                "categories": self.categories,
                "transcript_data": self.transcript_data
            }
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            if not silent:
                self.lbl_progress_status.setText(f"บันทึกโปรเจกต์แล้ว: {os.path.basename(file_path)}")
                QMessageBox.information(self, "บันทึกโปรเจกต์สำเร็จ", f"บันทึกข้อมูลไว้ที่:\n{file_path}")
        except Exception as e:
            if not silent:
                QMessageBox.critical(self, "บันทึกโปรเจกต์ไม่สำเร็จ", str(e))

    def save_project(self):
        if not self.transcript_data:
            return
        default_path = (
            os.path.splitext(self.audio_path)[0] + ".sermon.json"
            if self.audio_path else "project.sermon.json"
        )
        file_path, _ = QFileDialog.getSaveFileName(
            self, "บันทึกโปรเจกต์ที่ถอดความแล้ว", default_path, "Sermon Project (*.sermon.json *.json)"
        )
        if file_path:
            self.write_project_file(file_path, silent=False)

    def import_project(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "เลือกไฟล์โปรเจกต์ที่เคยถอดความไว้แล้ว", "", "Sermon Project (*.sermon.json *.json)"
        )
        if file_path:
            self.load_project_from_file(file_path)

    def load_project_from_file(self, file_path):
        try:
            self.stop_player()
            with open(file_path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            loaded_cats = payload.get("categories")
            if isinstance(loaded_cats, list):
                existing_ids = {c["id"] for c in self.categories}
                added = False
                for c in loaded_cats:
                    if isinstance(c, dict) and c.get("id") and c["id"] not in existing_ids:
                        self.categories.append(c)
                        existing_ids.add(c["id"])
                        added = True
                if added:
                    save_categories(self.categories)
                    self.refresh_category_combos()

            loaded_data = payload.get("transcript_data", [])
            if not isinstance(loaded_data, list):
                raise ValueError("รูปแบบไฟล์โปรเจกต์ไม่ถูกต้อง")

            self.transcript_data = loaded_data
            self.audio_duration = float(payload.get("audio_duration", 0.0))

            saved_audio = payload.get("audio_path")
            if saved_audio and os.path.exists(saved_audio):
                self.audio_path = saved_audio
                self.player.setSource(QUrl.fromLocalFile(saved_audio))
            else:
                if not self.audio_path or not os.path.exists(self.audio_path):
                    reply = QMessageBox.question(
                        self,
                        "ไม่พบไฟล์เสียงต้นฉบับในตำแหน่งเดิม",
                        "ต้องการเลือกไฟล์เสียงต้นฉบับของโปรเจกต์นี้ตอนนี้เลยไหมครับ? (เพื่อให้กดฟังและตัดเสียงได้)",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                    )
                    if reply == QMessageBox.StandardButton.Yes:
                        new_audio, _ = QFileDialog.getOpenFileName(
                            self, "เลือกไฟล์เสียงต้นฉบับ", "", "Audio Files (*.mp3 *.m4a *.wav)"
                        )
                        if new_audio:
                            self.audio_path = new_audio
                            self.player.setSource(QUrl.fromLocalFile(new_audio))

            fname = os.path.basename(self.audio_path) if self.audio_path else "(ยังไม่ได้เชื่อมต่อไฟล์เสียง)"
            self.lbl_file_info.setText(
                f"ไฟล์: {fname}\nความยาวรวม: {format_hms(self.audio_duration)} | ทั้งหมด {len(self.transcript_data)} ท่อน"
            )
            self.rebuild_table_from_data()
            self.progress_bar.setValue(100)
            self.lbl_progress_status.setText(f"โหลดโปรเจกต์สำเร็จ: {os.path.basename(file_path)}")

            has_audio = bool(self.audio_path and os.path.exists(self.audio_path))
            self.btn_transcribe.setEnabled(has_audio)
            self.set_player_controls_enabled(has_audio)
            self.btn_export_audio.setEnabled(has_audio and len(self.transcript_data) > 0)
            self.btn_export_text.setEnabled(len(self.transcript_data) > 0)
            self.btn_save_proj.setEnabled(len(self.transcript_data) > 0)
        except Exception as e:
            QMessageBox.critical(self, "Import ไม่สำเร็จ", f"ไม่สามารถเปิดไฟล์โปรเจกต์ได้:\n{e}")

    # ---------- การจัดการตาราง ----------
    def rebuild_table_from_data(self):
        self._is_updating_table = True
        self.table.setRowCount(len(self.transcript_data))

        for row, item in enumerate(self.transcript_data):
            cat = self.get_category_by_id(item.get("cat_id", self.categories[0]["id"]))
            item["cat_id"] = cat["id"]
            dur = max(0.0, item["end"] - item["start"])

            chk_item = QTableWidgetItem()
            chk_item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            chk_item.setCheckState(Qt.CheckState.Checked if item.get("checked") else Qt.CheckState.Unchecked)
            self.table.setItem(row, self.COL_CHECK, chk_item)

            no_item = QTableWidgetItem(str(row + 1))
            no_item.setFlags(no_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            no_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, self.COL_NO, no_item)

            start_item = QTableWidgetItem(format_hms(item["start"], show_decimal=True))
            self.table.setItem(row, self.COL_START, start_item)

            end_item = QTableWidgetItem(format_hms(item["end"], show_decimal=True))
            self.table.setItem(row, self.COL_END, end_item)

            dur_item = QTableWidgetItem(format_hms(dur, show_decimal=False))
            dur_item.setFlags(dur_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            dur_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, self.COL_DUR, dur_item)

            cat_item = QTableWidgetItem(cat["name"])
            cat_item.setFlags(cat_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            cat_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, self.COL_CAT, cat_item)

            text_item = QTableWidgetItem(item["text"])
            self.table.setItem(row, self.COL_TEXT, text_item)

            self.apply_row_color(row, cat["color"])

        self.adjust_time_columns_width()
        self.table.resizeRowsToContents()
        self._is_updating_table = False
        self.filter_table_by_search(self.txt_search.text())
        self.update_bottom_status()

    def apply_row_color(self, row, hex_color):
        bg_brush = QBrush(QColor(hex_color)) if hex_color else QBrush(QColor("#FFFFFF"))
        fg_brush = QBrush(get_contrast_text_color(hex_color))
        for col in range(self.table.columnCount()):
            cell = self.table.item(row, col)
            if cell:
                cell.setBackground(bg_brush)
                cell.setForeground(fg_brush)

    def filter_table_by_search(self, query):
        q = query.strip().lower()
        for row, item in enumerate(self.transcript_data):
            if not q:
                self.table.setRowHidden(row, False)
            else:
                cat = self.get_category_by_id(item["cat_id"])
                match = (q in item["text"].lower()) or (q in cat["name"].lower())
                self.table.setRowHidden(row, not match)

    def get_padded_bounds(self, item):
        start = max(0.0, float(item["start"]))
        end = float(item["end"])
        if self.audio_duration > 0:
            end = min(end, self.audio_duration)
        end = max(start + 0.05, end)
        return start, end

    def update_bottom_status(self):
        total_rows = len(self.transcript_data)
        checked_count = 0
        checked_duration = 0.0
        for item in self.transcript_data:
            if item.get("checked"):
                checked_count += 1
                s, e = self.get_padded_bounds(item)
                checked_duration += max(0.0, e - s)
        self.lbl_bottom_summary.setText(
            f"เลือกแล้ว: {checked_count} / {total_rows} ท่อน ({format_hms(checked_duration)})"
        )

    def set_all_checks(self, checked):
        rows = [r for r in range(len(self.transcript_data)) if not self.table.isRowHidden(r)]
        self._is_updating_table = True
        for r in rows:
            self.transcript_data[r]["checked"] = checked
            cell = self.table.item(r, self.COL_CHECK)
            if cell:
                cell.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
        self._is_updating_table = False
        self.update_bottom_status()

    def invert_checks(self):
        self._is_updating_table = True
        for row in range(len(self.transcript_data)):
            if self.table.isRowHidden(row):
                continue
            new_val = not self.transcript_data[row].get("checked", False)
            self.transcript_data[row]["checked"] = new_val
            cell = self.table.item(row, self.COL_CHECK)
            if cell:
                cell.setCheckState(Qt.CheckState.Checked if new_val else Qt.CheckState.Unchecked)
        self._is_updating_table = False
        self.update_bottom_status()

    def check_rows_by_category(self):
        target_cat_id = self.combo_batch_cat.currentData()
        if not target_cat_id:
            return
        self._is_updating_table = True
        for row, item in enumerate(self.transcript_data):
            is_match = (item.get("cat_id") == target_cat_id)
            item["checked"] = is_match
            cell = self.table.item(row, self.COL_CHECK)
            if cell:
                cell.setCheckState(Qt.CheckState.Checked if is_match else Qt.CheckState.Unchecked)
        self._is_updating_table = False
        self.update_bottom_status()

    def merge_checked_rows(self):
        checked_indices = [i for i, item in enumerate(self.transcript_data) if item.get("checked")]
        if len(checked_indices) < 2:
            QMessageBox.information(self, "รวมท่อน", "กรุณาติ๊กเลือกอย่างน้อย 2 ท่อนเพื่อรวมเป็นท่อนเดียวครับ")
            return
        first_idx = checked_indices[0]
        merged_start = min(self.transcript_data[i]["start"] for i in checked_indices)
        merged_end = max(self.transcript_data[i]["end"] for i in checked_indices)
        merged_text = " ".join(self.transcript_data[i]["text"] for i in checked_indices if self.transcript_data[i]["text"])
        first_cat = self.transcript_data[first_idx]["cat_id"]

        new_item = {
            "start": merged_start,
            "end": merged_end,
            "text": merged_text,
            "cat_id": first_cat,
            "checked": True
        }
        for i in reversed(checked_indices):
            self.transcript_data.pop(i)
        self.transcript_data.insert(first_idx, new_item)
        self.rebuild_table_from_data()

    def add_custom_segment(self):
        """เพิ่มท่อนเนื้อเรื่องด้วยตัวเอง โดยอิงเวลาจากแถวที่เลือกอยู่หรือเวลาปัจจุบันใน Media Player"""
        default_start = 0.0
        insert_pos = len(self.transcript_data)

        selected_rows = sorted(set(idx.row() for idx in self.table.selectedIndexes() if 0 <= idx.row() < len(self.transcript_data)))
        if selected_rows:
            last_sel = selected_rows[-1]
            insert_pos = last_sel + 1
            default_start = float(self.transcript_data[last_sel]["end"])
        elif 0 <= self.current_row < len(self.transcript_data):
            insert_pos = self.current_row + 1
            default_start = float(self.transcript_data[self.current_row]["end"])
        elif self.player.position() > 0:
            default_start = round(self.player.position() / 1000.0, 2)

        default_end = round(default_start + 10.0, 2)
        if self.audio_duration > 0:
            default_start = min(default_start, max(0.0, self.audio_duration - 1.0))
            default_end = min(self.audio_duration, max(default_start + 1.0, default_end))

        target_cat_id = self.combo_batch_cat.currentData() or self.categories[0]["id"]
        new_item = {
            "start": round(default_start, 2),
            "end": round(default_end, 2),
            "text": "(ท่อนที่เพิ่มเอง — กด F2 เพื่อแก้เวลาและข้อความ)",
            "cat_id": target_cat_id,
            "checked": True
        }
        self.transcript_data.insert(insert_pos, new_item)
        self.rebuild_table_from_data()
        self.table.selectRow(insert_pos)
        self.table.scrollToItem(self.table.item(insert_pos, self.COL_NO))
        self.current_row = insert_pos

        has_audio = bool(self.audio_path and os.path.exists(self.audio_path))
        self.btn_export_audio.setEnabled(has_audio)
        self.btn_export_text.setEnabled(True)
        self.btn_save_proj.setEnabled(True)
        self.lbl_progress_status.setText(
            f"เพิ่มท่อนที่ #{insert_pos + 1} ({format_hms(new_item['start'], True)} - {format_hms(new_item['end'], True)}) แล้ว — แก้เวลาได้โดยตรงในตาราง"
        )

    def _get_target_rows_for_time_edit(self):
        selected_rows = sorted(set(
            idx.row() for idx in self.table.selectedIndexes()
            if 0 <= idx.row() < len(self.transcript_data)
        ))
        if selected_rows:
            return selected_rows
        if 0 <= self.current_row < len(self.transcript_data):
            return [self.current_row]
        return []

    def _parse_step_seconds(self, widget):
        thai_digits = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
        raw = widget.text().strip().translate(thai_digits)
        try:
            val = abs(float(raw))
            if val <= 0:
                val = 1.0
                widget.setText("1")
            return val
        except ValueError:
            widget.setText("1")
            return 1.0

    def adjust_selected_time(self, field, direction):
        """นำค่าที่ฟิกซ์ไว้ในช่อง เริ่ม/จบ ไปบวก (+) หรือลบ (-) กับเวลาของท่อนที่เลือกอยู่ทันที"""
        step = self._parse_step_seconds(self.txt_step_start if field == "start" else self.txt_step_end)
        delta = round(step * direction, 2)
        target_rows = self._get_target_rows_for_time_edit()
        if not target_rows:
            QMessageBox.information(self, "ยังไม่ได้เลือกท่อน", "กรุณาคลิกเลือกแถวในตารางก่อนกดปุ่ม + หรือ - ครับ")
            return

        self._is_updating_table = True
        for r in target_rows:
            item = self.transcript_data[r]
            if field == "start":
                updated_start = max(0.0, round(float(item["start"]) + delta, 2))
                if updated_start > float(item["end"]) - 0.05:
                    updated_start = max(0.0, round(float(item["end"]) - 0.05, 2))
                item["start"] = updated_start
                if self.table.item(r, self.COL_START):
                    self.table.item(r, self.COL_START).setText(format_hms(item["start"], show_decimal=True))
            else:
                updated_end = max(float(item["start"]) + 0.05, round(float(item["end"]) + delta, 2))
                if self.audio_duration > 0:
                    updated_end = min(updated_end, round(self.audio_duration, 2))
                item["end"] = updated_end
                if self.table.item(r, self.COL_END):
                    self.table.item(r, self.COL_END).setText(format_hms(item["end"], show_decimal=True))

            dur = max(0.0, float(item["end"]) - float(item["start"]))
            if self.table.item(r, self.COL_DUR):
                self.table.item(r, self.COL_DUR).setText(format_hms(dur, show_decimal=False))

        self._is_updating_table = False
        self.update_bottom_status()
        action_word = "เพิ่ม" if direction > 0 else "ลด"
        field_label = "เริ่ม" if field == "start" else "จบ"
        step_disp = int(step) if step.is_integer() else step
        self.lbl_progress_status.setText(
            f"{action_word}เวลา [{field_label}] {step_disp} วิ ให้ท่อนที่เลือก ({len(target_rows)} ท่อน) แล้ว"
        )

    def delete_selected_segments(self):
        if not self.transcript_data:
            return
        rows_to_del = sorted(set(
            idx.row() for idx in self.table.selectedIndexes()
            if 0 <= idx.row() < len(self.transcript_data)
        ))
        if not rows_to_del and 0 <= self.current_row < len(self.transcript_data):
            rows_to_del = [self.current_row]
        if not rows_to_del:
            rows_to_del = [i for i, item in enumerate(self.transcript_data) if item.get("checked")]
        if not rows_to_del:
            QMessageBox.information(self, "ยังไม่ได้เลือกท่อน", "กรุณาเลือกท่อนเนื้อเรื่องที่ต้องการลบก่อนครับ")
            return

        for r in reversed(rows_to_del):
            self.transcript_data.pop(r)

        self.rebuild_table_from_data()
        next_row = min(rows_to_del[0], len(self.transcript_data) - 1)
        if next_row >= 0:
            self.table.selectRow(next_row)
            self.current_row = next_row
        else:
            self.current_row = -1
            self.btn_export_audio.setEnabled(False)
            self.btn_export_text.setEnabled(False)
            self.btn_save_proj.setEnabled(False)

        self.lbl_progress_status.setText(f"ลบท่อนเนื้อเรื่องออก {len(rows_to_del)} ท่อนแล้ว")

    def on_table_cell_clicked(self, row, column):
        if 0 <= row < len(self.transcript_data):
            self.current_row = row

    def on_table_double_clicked(self, row, column):
        if 0 <= row < len(self.transcript_data):
            self.current_row = row
            self.apply_check_and_color([row], True)

    def on_item_changed(self, item):
        if self._is_updating_table:
            return
        row = item.row()
        col = item.column()
        if not (0 <= row < len(self.transcript_data)):
            return

        if col == self.COL_CHECK:
            is_checked = (item.checkState() == Qt.CheckState.Checked)
            self.transcript_data[row]["checked"] = is_checked
            self.update_bottom_status()

        elif col in (self.COL_START, self.COL_END):
            val = parse_time_str(item.text())
            self._is_updating_table = True
            key = "start" if col == self.COL_START else "end"
            if val is not None:
                self.transcript_data[row][key] = round(val, 2)
                item.setText(format_hms(val, show_decimal=True))
                dur = max(0.0, self.transcript_data[row]["end"] - self.transcript_data[row]["start"])
                self.table.item(row, self.COL_DUR).setText(format_hms(dur, show_decimal=False))
            else:
                item.setText(format_hms(self.transcript_data[row][key], show_decimal=True))
            self._is_updating_table = False
            self.update_bottom_status()

        elif col == self.COL_TEXT:
            self.transcript_data[row]["text"] = item.text()

    # ---------- มีเดียเพลย์เยอร์: พรีวิวฟังท่อนที่เลือกต่อๆ กัน ----------
    def play_checked_sequentially(self):
        if not self.audio_path or not self.transcript_data:
            return
        queue = [i for i, item in enumerate(self.transcript_data) if item.get("checked")]
        if not queue:
            QMessageBox.information(self, "ยังไม่ได้ติ๊กเลือกท่อน", "กรุณาติ๊กเลือกท่อนที่ต้องการฟังทั้งหมดก่อนครับ")
            return

        self.play_mode_label = "ที่ติ๊กทั้งหมด"
        self.play_queue = queue
        self.play_queue_idx = 0
        self._start_queue_item(0)

    def play_selected_rows_sequentially(self):
        if not self.audio_path or not self.transcript_data:
            return
        queue = sorted(set(
            idx.row() for idx in self.table.selectedIndexes()
            if 0 <= idx.row() < len(self.transcript_data)
        ))
        if not queue and 0 <= self.current_row < len(self.transcript_data):
            queue = [self.current_row]
        if not queue:
            QMessageBox.information(
                self,
                "ยังไม่ได้เลือกท่อน",
                "กรุณาคลิกเลือก หรือกด Shift/Ctrl ไฮไลต์ท่อนที่ต้องการฟังในตารางก่อนครับ"
            )
            return

        self.play_mode_label = "เฉพาะที่เลือกอยู่"
        self.play_queue = queue
        self.play_queue_idx = 0
        self._start_queue_item(0)

    def _start_queue_item(self, q_idx):
        if not (0 <= q_idx < len(self.play_queue)):
            self.stop_player(finished_msg="เล่นครบทุกท่อนที่เลือกแล้ว")
            return

        self.play_queue_idx = q_idx
        row_idx = self.play_queue[q_idx]
        item = self.transcript_data[row_idx]
        s, e = self.get_padded_bounds(item)

        self.active_seg_start_ms = int(s * 1000)
        self.active_seg_end_ms = int(e * 1000)

        self.table.selectRow(row_idx)
        self.table.scrollToItem(self.table.item(row_idx, self.COL_NO))
        self.current_row = row_idx

        preview_text = item["text"][:45] + ("..." if len(item["text"]) > 45 else "")
        mode_prefix = f"[{self.play_mode_label}] " if self.play_mode_label else ""
        self.lbl_player_now.setText(
            f"▶ {mode_prefix}ท่อนที่ {q_idx + 1} / {len(self.play_queue)} (ลำดับ #{row_idx + 1}): {preview_text}"
        )

        self.player.setPosition(self.active_seg_start_ms)
        self.player.play()

    def play_prev_in_queue(self):
        if self.play_queue and self.play_queue_idx > 0:
            self._start_queue_item(self.play_queue_idx - 1)
        elif self.play_queue:
            self._start_queue_item(0)

    def play_next_in_queue(self):
        if self.play_queue and self.play_queue_idx + 1 < len(self.play_queue):
            self._start_queue_item(self.play_queue_idx + 1)

    def toggle_pause_resume(self):
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
            self.lbl_player_now.setText("⏸ พักการเล่นชั่วคราว")
        elif self.player.playbackState() == QMediaPlayer.PlaybackState.PausedState:
            self.player.play()
        elif self.play_queue:
            self._start_queue_item(max(0, self.play_queue_idx))
        else:
            self.play_checked_sequentially()

    def stop_player(self, finished_msg=None):
        self.player.stop()
        self.play_queue = []
        self.play_queue_idx = -1
        self.active_seg_start_ms = 0
        self.active_seg_end_ms = 0
        self.lbl_player_now.setText(
            finished_msg if finished_msg else "⏹ หยุดเล่นแล้ว"
        )

    def on_player_duration_changed(self, duration_ms):
        if duration_ms > 0:
            self.slider_pos.setRange(0, duration_ms)
            self.lbl_time_tot.setText(format_hms(duration_ms / 1000.0))
            if self.audio_duration <= 0:
                self.audio_duration = duration_ms / 1000.0

    def on_player_position_changed(self, position_ms):
        if not self._is_slider_dragging:
            self.slider_pos.setValue(position_ms)
        self.lbl_time_cur.setText(format_hms(position_ms / 1000.0))

        if self.play_queue and 0 <= self.play_queue_idx < len(self.play_queue):
            if self.active_seg_end_ms > 0 and position_ms >= self.active_seg_end_ms:
                next_idx = self.play_queue_idx + 1
                if next_idx < len(self.play_queue):
                    self._start_queue_item(next_idx)
                else:
                    self.stop_player(finished_msg="✓ พรีวิวเล่นครบทุกท่อนที่เลือกแล้ว")

    def on_slider_pressed(self):
        self._is_slider_dragging = True

    def on_slider_released(self):
        self._is_slider_dragging = False
        self.player.setPosition(self.slider_pos.value())

    # ---------- Export เสียง & ข้อความ ----------
    def choose_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "เลือกโฟลเดอร์ปลายทาง", self.txt_output_folder.text())
        if folder:
            self.output_folder = folder
            self.txt_output_folder.setText(folder)

    def export_selected_audio(self):
        if not self.audio_path or not self.transcript_data:
            return

        selected_segments = [item for item in self.transcript_data if item.get("checked")]
        if not selected_segments:
            highlighted_rows = sorted(set(idx.row() for idx in self.table.selectedIndexes()))
            selected_segments = [
                self.transcript_data[r] for r in highlighted_rows if 0 <= r < len(self.transcript_data)
            ]
        if not selected_segments:
            QMessageBox.warning(self, "ยังไม่ได้เลือกท่อน", "กรุณาคลิกเลือกหรือติ๊กเลือกท่อนที่ต้องการตัด/รวมก่อนครับ")
            return

        ffmpeg_exe = get_ffmpeg_executable()
        if not ffmpeg_exe:
            QMessageBox.critical(
                self,
                "ไม่พบเครื่องมือตัดต่อเสียง (FFmpeg)",
                "กรุณาติดตั้งไลบรารี imageio-ffmpeg ก่อนโดยเปิด CMD แล้วพิมพ์:\n\npip install imageio-ffmpeg"
            )
            return

        fmt = "mp3" if self.rb_mp3.isChecked() else ("m4a" if self.rb_m4a.isChecked() else "wav")
        mode = "merge" if self.rb_merge.isChecked() else ("split" if self.rb_split.isChecked() else "both")
        out_dir = self.txt_output_folder.text().strip() or self.output_folder
        base_name = os.path.splitext(os.path.basename(self.audio_path))[0]

        self.btn_export_audio.setEnabled(False)
        self.btn_stop_export.setEnabled(True)
        self.lbl_progress_status.setText("กำลังเริ่มตัดต่อไฟล์เสียง...")

        self.audio_export_thread = AudioExportThread(
            ffmpeg_exe,
            self.audio_path,
            selected_segments,
            out_dir,
            base_name,
            fmt,
            mode,
            pad_start=0.0,
            pad_end=0.0,
            max_duration=self.audio_duration if self.audio_duration > 0 else None,
        )
        self.audio_export_thread.progress.connect(lambda msg: self.lbl_progress_status.setText(msg))
        self.audio_export_thread.finished.connect(self.on_audio_export_done)
        self.audio_export_thread.error.connect(self.on_audio_export_error)
        self.audio_export_thread.start()

    def stop_audio_export(self):
        if hasattr(self, "audio_export_thread") and self.audio_export_thread.isRunning():
            self.audio_export_thread.stop()
            self.lbl_progress_status.setText("กำลังยกเลิกการตัดต่อ...")

    def on_audio_export_done(self, info_msg):
        self.btn_export_audio.setEnabled(True)
        self.btn_stop_export.setEnabled(False)
        self.lbl_progress_status.setText("ตัดต่อและบันทึกไฟล์เสียงเสร็จสิ้น!")
        QMessageBox.information(self, "Export สำเร็จ", f"บันทึกไฟล์เสียงเรียบร้อยแล้ว:\n\n{info_msg}")

    def on_audio_export_error(self, err_msg):
        self.btn_export_audio.setEnabled(True)
        self.btn_stop_export.setEnabled(False)
        self.lbl_progress_status.setText("การตัดต่อหยุดลงหรือเกิดข้อผิดพลาด")
        QMessageBox.warning(self, "แจ้งเตือน", err_msg)

    def export_transcript(self):
        if not self.transcript_data:
            return
        default_name = os.path.splitext(os.path.basename(self.audio_path))[0] if self.audio_path else "transcript"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "บันทึกบทถอดความ",
            default_name,
            "HTML มีสีไฮไลต์ เปิดใน Word ได้ (*.html);;Excel CSV (*.csv);;Text File (*.txt);;Subtitle (*.srt)"
        )
        if not file_path:
            return

        try:
            ext = os.path.splitext(file_path)[1].lower()
            if ext == ".csv":
                with open(file_path, "w", encoding="utf-8-sig", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["ลำดับ", "เวลาเริ่ม", "เวลาจบ", "ความยาว", "ชนิด", "ข้อความคำพูด"])
                    for idx, r in enumerate(self.transcript_data, start=1):
                        cat = self.get_category_by_id(r["cat_id"])
                        dur = max(0.0, r["end"] - r["start"])
                        writer.writerow([
                            idx, format_hms(r["start"], True), format_hms(r["end"], True),
                            format_hms(dur), cat["name"], r["text"]
                        ])
            elif ext == ".txt":
                with open(file_path, "w", encoding="utf-8") as f:
                    for idx, r in enumerate(self.transcript_data, start=1):
                        cat = self.get_category_by_id(r["cat_id"])
                        f.write(
                            f"{idx}. [{format_hms(r['start'], True)} - {format_hms(r['end'], True)}] "
                            f"({cat['name']}) {r['text']}\n"
                        )
            elif ext == ".srt":
                with open(file_path, "w", encoding="utf-8") as f:
                    for idx, r in enumerate(self.transcript_data, start=1):
                        f.write(f"{idx}\n{format_srt_time(r['start'])} --> {format_srt_time(r['end'])}\n{r['text']}\n\n")
            else:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write("<!DOCTYPE html><html lang='th'><head><meta charset='UTF-8'>\n")
                    f.write(f"<title>{html.escape(default_name)}</title>\n")
                    f.write("<style>body{font-family:'Tahoma','Sarabun',sans-serif;font-size:16px;margin:20px;}"
                            "table{width:100%;border-collapse:collapse;}th,td{border:1px solid #ccc;padding:8px;text-align:left;}"
                            "th{background:#f1f5f9;}</style></head><body>\n")
                    f.write(f"<h2>{html.escape(default_name)}</h2>\n")
                    f.write("<table><thead><tr><th>ลำดับ</th><th>เวลาเริ่ม</th><th>เวลาจบ</th><th>ความยาว</th><th>ชนิด</th><th>ข้อความคำพูด</th></tr></thead><tbody>\n")
                    for idx, r in enumerate(self.transcript_data, start=1):
                        cat = self.get_category_by_id(r["cat_id"])
                        dur = max(0.0, r["end"] - r["start"])
                        style = ""
                        if cat["color"]:
                            tc = get_contrast_text_color(cat["color"]).name()
                            style = f" style='background:{cat['color']};color:{tc};'"
                        f.write(
                            f"<tr{style}><td>{idx}</td><td>{format_hms(r['start'], True)}</td>"
                            f"<td>{format_hms(r['end'], True)}</td><td>{format_hms(dur)}</td>"
                            f"<td>{html.escape(cat['name'])}</td><td>{html.escape(r['text'])}</td></tr>\n"
                        )
                    f.write("</tbody></table></body></html>")
            QMessageBox.information(self, "บันทึกสำเร็จ", f"บันทึกบทถอดความเรียบร้อยแล้วที่:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "บันทึกไม่สำเร็จ", str(e))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Tahoma", 10))
    window = SermonTranscriberApp()
    window.show()
    sys.exit(app.exec())