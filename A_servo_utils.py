import numpy as np

# === 서보 기준(중립 90/90) & 부호/스케일 ===
BASE_YAW_DEG   = 90.0   # 서보 중립
BASE_PITCH_DEG = 90.0   # 서보 중립
YAW_SIGN       = -1.0   # 반대로 가면 -1.0
PITCH_SIGN     = +1.0   # 반대로 가면 -1.0
YAW_SCALE      = 1.0    # 필요시 감도 미세조정
PITCH_SCALE    = 1.0

def send_servo_angles(ctl, yaw_cmd, pitch_cmd):
    try:
        print(f"[Servo] send: yaw={yaw_cmd:.2f}°, pitch={pitch_cmd:.2f}°")
        ctl.set_angles(pitch_cmd, yaw_cmd)  # (pitch, yaw) 순서
    except Exception as e:
        print(f"[Servo ERROR] {e}")

        
def to_servo_cmd(yaw_opt_deg, pitch_opt_deg):
    """
    광학각(카메라 전방 +Z 기준의 yaw/pitch, 단위 °) -> 서보 명령각(°)
    '90/90이 정면'이 되도록 중립 오프셋을 더해준다.
    """
    y = BASE_YAW_DEG   + YAW_SIGN   * (YAW_SCALE   * yaw_opt_deg)
    p = BASE_PITCH_DEG + PITCH_SIGN * (PITCH_SCALE * pitch_opt_deg)
    # 안전 클램프(필요하면 유지/수정)
    y = max(0.0, min(180.0, y))
    p = max(0.0, min(180.0, p))
    return y, p

def yaw_pitch_from_X(X, O, y_up_is_negative=True):
    v = X - O
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    yaw   = np.degrees(np.arctan2(vx, vz))
    pitch = np.degrees(np.arctan2((-vy if y_up_is_negative else vy), np.hypot(vx, vz)))
    return yaw, pitch