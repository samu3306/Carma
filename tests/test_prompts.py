from src.vlm.prompts import angle_quality_prompt, plate_visibility_prompt


def test_interior_front_prompt_requires_visible_seat_area():
    prompt = angle_quality_prompt("interior_front")
    assert "至少一張前座" in prompt
    assert "只拍方向盤" in prompt
    assert "quality_passed 必須為 false" in prompt


def test_prompt_never_reveals_selected_expected_view():
    prompt = angle_quality_prompt("left_rear")
    assert "使用者指定本張照片" not in prompt
    assert "左右一律以車輛自身方向為準" in prompt
    assert "必須先獨立判定 detected_view" in prompt
    assert "right_rear 檢核標準" in prompt


def test_right_rear_prompt_requires_actual_rear_plate():
    prompt = angle_quality_prompt("right_rear")
    assert "車尾車牌" in prompt
    assert "車牌在畫面外" in prompt
    assert "不得因看見車尾或車身就假設車牌存在" in prompt


def test_plate_prompt_ignores_background_vehicle_plates():
    prompt = plate_visibility_prompt("left_rear")
    assert "主要目標車" in prompt
    assert "背景、旁邊或遠處其他車輛的車牌一律忽略" in prompt
    assert "車尾車牌" in prompt
    assert "plate_text" in prompt
    assert "不可猜測" in prompt
