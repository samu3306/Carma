"""Prompt additions for human-reviewed angle demonstrations."""

FEW_SHOT_INTRO = """以下圖片是人工審核確認的拍攝角度範例，只用來理解角度分類，不代表待判斷照片的品質。
左右方向以車輛自身方向為準，不以觀看者畫面左右為準。
如果新照片主要是正前方、正後方，或無法可靠判斷左右，detected_view 必須輸出 other，不可輸出 front 或 rear。"""

TARGET_INTRO = "以下是需要獨立判斷的新照片。不可將任何人工範例標籤直接套用到新照片："


REAR_SIDE_INTRO = """你是車尾左右視角二分類器。以下人工照片的標籤是權威答案。
請比較目標照片與成對範例的車尾幾何及可見車身側面，不可使用檔名，也不可處理照片品質。
本任務只能選 left_rear 或 right_rear；不確定時降低 confidence，但仍選較相似的一類。"""

REAR_SIDE_TARGET_PROMPT = """只判斷這張目標照片較符合 left_rear 或 right_rear。
只輸出合法 JSON，並固定使用以下欄位；除 detected_view、confidence、explanation 外，其餘值不得修改：
{
  "detected_view": "left_rear",
  "vehicle_visible": true,
  "plate_visible": true,
  "quality_passed": true,
  "quality_issues": [],
  "confidence": 0.8,
  "explanation": "簡短說明可見的是車輛自身哪一側",
  "retake_instruction": null
}"""
