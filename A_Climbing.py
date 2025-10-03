import time
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import csv
import argparse

# === MediaPipe 모듈 ===
from Climb_Mediapipe import PoseTracker, TouchCounter, draw_pose_points

# 레이저 찾기
from find_laser import capture_once_and_return

# 삼각측량법 관련 코드
from A_stereo_utils import load_stereo, rectify, triangulate_xy, raw_to_rectified_point

# hold 관련 코드
from A_hold_utils import extract_holds_with_indices, merge_holds_by_center, assign_indices

# servo 관련 코드
from A_servo_utils import send_servo_angles, to_servo_cmd, yaw_pitch_from_X

# === (NEW) 웹 모듈 - 색상 선택 ===
_USE_WEB = True
try:
    from A_web import choose_color_via_web
except Exception:
    _USE_WEB = False
    def choose_color_via_web(*a, **k):
        raise RuntimeError("color_web 모듈(A_web)이 로드되지 않았습니다.")

# ========= 사용자 환경 경로 =========
# NPZ_PATH       = "C:\Users\PC\Desktop\Segmentation_Hold\stereo_params_scaled.npz"
# MODEL_PATH     = "C:\Users\PC\Desktop\Segmentation_Hold\best_5.pt"

NPZ_PATH       = r"/home/jsh/Desktop/1003/stereo_params_scaled.npz"
MODEL_PATH     = r"/home/jsh/Desktop/1003/best_5.pt"

# 젯슨나노에서 1, 0으로 사용
CAM1_INDEX     = 1   # 왼쪽 카메라
CAM2_INDEX     = 0   # 오른쪽 카메라

SWAP_DISPLAY   = False   # 화면 표시 좌/우 스와프

WINDOW_NAME    = "Rectified L | R"
THRESH_MASK    = 0.7
ROW_TOL_Y      = 30
SELECTED_COLOR = None    # 예: 'orange' (None=전체)

# 자동 진행(터치→다음 홀드) 관련
TOUCH_THRESHOLD = 10     # in-polygon 연속 프레임 임계(기본 10)
ADV_COOLDOWN    = 0.5    # 연속 넘김 방지 쿨다운(sec)

# 저장 옵션
SAVE_VIDEO     = False
OUT_FPS        = 30
OUT_PATH       = "stereo_overlay.mp4"
CSV_GRIPS_PATH = "grip_records.csv"

# 런타임 보정 오프셋(레이저 실측)
CAL_YAW_OFFSET   = 0.0
CAL_PITCH_OFFSET = 0.0

# ---- 레이저 원점(LEFT 기준) 오프셋 (cm) ----
LASER_OFFSET_CM_LEFT = 1.15
LASER_OFFSET_CM_UP   = 5.2
LASER_OFFSET_CM_FWD  = -0.6
Y_UP_IS_NEGATIVE     = True  # 위 방향이 -y인 좌표계면 True

# === 서보 기준(중립 90/90) & 부호/스케일 ===
BASE_YAW_DEG   = 90.0   # 서보 중립
BASE_PITCH_DEG = 90.0   # 서보 중립
YAW_SIGN       = -1.0   # 반대로 가면 -1.0
PITCH_SIGN     = +1.0   # 반대로 가면 -1.0
YAW_SCALE      = 1.0    # 필요시 감도 미세조정
PITCH_SCALE    = 1.0

# ======== Servo controller import (stub fallback) ========
try:
    from servo_control import DualServoController
    HAS_SERVO = True
except Exception:
    HAS_SERVO = False
    class DualServoController:
        def __init__(self, *a, **k): print("[Servo] (stub) controller unavailable")
        def set_angles(self, pitch=None, yaw=None): print(f"[Servo] (stub) set_angles: P={pitch}, Y={yaw}")
        def center(self): print("[Servo] (stub) center")
        def query(self): print("[Servo] (stub) query"); return ""
        def laser_on(self): print("[Servo] (stub) laser_on")
        def laser_off(self): print("[Servo] (stub) laser_off")
        def close(self): pass
# ======================

def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--no_auto_advance", action="store_true")
    ap.add_argument("--no_web", action="store_true")
    return ap.parse_args()

def _verify_paths():
    for p in (NPZ_PATH, MODEL_PATH):
        if not Path(p).exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {p}")

def open_cams(idx1, idx2, size):
    W, H = size
    cap1 = cv2.VideoCapture(idx1, cv2.CAP_V4L2)
    cap2 = cv2.VideoCapture(idx2, cv2.CAP_V4L2)
    cap1.set(cv2.CAP_PROP_FRAME_WIDTH,  W); cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap2.set(cv2.CAP_PROP_FRAME_WIDTH,  W); cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    if not cap1.isOpened() or not cap2.isOpened():
        raise SystemExit("카메라 오픈 실패. 연결/권한 확인.")
    return cap1, cap2

def imshow_scaled(win, img, maxw=None):
    if not maxw:
        cv2.imshow(win, img)
        return
    h, w = img.shape[:2]
    if w > maxw:
        s = maxw / w
        img = cv2.resize(img, (int(w*s), int(h*s)))
    cv2.imshow(win, img)

def xoff_for(side, W, swap):
    return (W if swap else 0) if side=="L" else (0 if swap else W)

def angle_between(v1, v2):
    a = np.linalg.norm(v1); b = np.linalg.norm(v2)
    if a == 0 or b == 0: return 0.0
    cosang = np.clip(np.dot(v1, v2) / (a * b), -1.0, 1.0)
    return np.degrees(np.arccos(cosang))

def wrap_deg(d):
    return (d + 180.0) % 360.0 - 180.0

# ==== 색상 맵 & 선택 ====

ALL_COLORS = {
    'red':'Hold_Red','orange':'Hold_Orange','yellow':'Hold_Yellow','green':'Hold_Green',
    'blue':'Hold_Blue','purple':'Hold_Purple','pink':'Hold_Pink','white':'Hold_White',
    'black':'Hold_Black','gray':'Hold_Gray','lime':'Hold_Lime','sky':'Hold_Sky',
}

def _sanitize_label(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum() or ch in ("_", "-"))

def ask_color_and_map_to_class(all_colors_dict):
    print("가능한 색상:", ", ".join(all_colors_dict.keys()))
    s = input("필터할 색상 입력(엔터=전체): ").strip().lower()
    if not s:
        print("→ 전체 표시 사용")
        return None, "all"   # (모델클래스=None, 파일라벨="all")
    mapped = all_colors_dict.get(s)
    if mapped is None:
        print(f"입력 '{s}' 은(는) 유효하지 않은 색입니다. 전체 표시 사용")
        return None, "all"
    print(f"선택된 클래스명: {mapped}")
    return mapped, s        # (모델클래스, 파일라벨)

# === (NEW) CSV에서 경로 순서 로드 ===
def load_route_ids_from_csv(path):
    route_ids = []
    try:
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "hold_id" in row:
                    try:
                        hid = int(row["hold_id"])
                        route_ids.append(hid)
                    except:
                        pass
    except FileNotFoundError:
        print(f"[Warn] 경로 CSV '{path}' 없음 → 인덱스 순서 사용")
    return route_ids

# ======================
# main()에서 호출되는 단계 함수
# ======================

def _load_stereo_and_log():
    map1x, map1y, map2x, map2y, P1, P2, size, B, M = load_stereo(NPZ_PATH)
    W, H = size
    print(f"[Info] image_size={(W,H)}, baseline~{B:.2f} mm")
    return map1x, map1y, map2x, map2y, P1, P2, size, (W, H)

def _capture_laser_and_rectify(args, W, H):
    # ===== (NEW) 레이저 좌표 먼저 측정 (find_laser) =====
    try:
        laser_raw = capture_once_and_return(
            port=args.port,
            baud=args.baud,
            wait_s=2.0,
            settle_n=8,
            show_preview=True,           # 히트맵 확인하고 싶으면 True
            center_pitch=90.0,           # ← 필수: 시작 90/90
            center_yaw=90.0,
            servo_settle_s=0.5
        )
    except Exception as e:
        print(f"[A_Climbing] find_laser error: {e} → continue without laser")
        laser_raw = None

    if laser_raw is None:
        print("[A_Climbing] 레이저 좌표 취득 실패(취소/에러). 계속 진행.")
        return None

    # 원본 좌표(보정 전)
    cam0_raw = laser_raw["cam0"]  # 보통 LEFT(=CAM1_INDEX=1)
    cam1_raw = laser_raw["cam1"]  # 보통 RIGHT(=CAM2_INDEX=2)

    # npz에서 내부/왜곡/정렬행렬을 꺼내야 함
    S = np.load(NPZ_PATH, allow_pickle=True)
    K1, D1, R1, P1_ = S["K1"], S["D1"], S["R1"], S["P1"]
    K2, D2, R2, P2_ = S["K2"], S["D2"], S["R2"], S["P2"]

    # 원본→레티파이 좌표로 변환
    camL_rect = raw_to_rectified_point(cam0_raw, K1, D1, R1, P1_) if cam0_raw else None
    camR_rect = raw_to_rectified_point(cam1_raw, K2, D2, R2, P2_) if cam1_raw else None

    laser_px = {
        "left_rect":  camL_rect,   # Lr 좌표계
        "right_rect": camR_rect,   # Rr 좌표계
        "image_size": (W, H),
    }
    print(f"[A_Climbing] 레이저(원본): L={cam0_raw}, R={cam1_raw}")
    print(f"[A_Climbing] 레이저(레티파이): L={camL_rect}, R={camR_rect}")
    return laser_px

def _compute_laser_origin_left():
    # 레이저 원점 O (LEFT 기준)
    L = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    dx = -LASER_OFFSET_CM_LEFT * 10.0
    dy = (-1.0 if Y_UP_IS_NEGATIVE else 1.0) * LASER_OFFSET_CM_UP * 10.0
    dz = LASER_OFFSET_CM_FWD * 10.0
    O  = L + np.array([dx, dy, dz], dtype=np.float64)
    print(f"[Laser] Origin O (mm, LEFT-based) = {O}")
    return L, O

def _choose_color(args):
    # ================= 색상 필터 선택 =================
    selected_class_name  = None     # 모델 클래스명 (예: Hold_Green)
    selected_color_label = "all"    # 파일명 라벨 (예: green, orange, all)

    # 1) 웹에서 색상 선택(가능하고 --no_web가 아닐 때)
    if (not args.no_web) and _USE_WEB:
        try:
            chosen = choose_color_via_web(
                all_colors=list(ALL_COLORS.keys()),
                defaults={"port": args.port, "baud": args.baud}
            )  # ""이면 전체
            if chosen:
                mapped = ALL_COLORS.get(chosen)
                if mapped is None:
                    print(f"[Filter] 웹 선택 '{chosen}' 무효 → 전체 표시")
                else:
                    print(f"[Filter] 웹 선택: {chosen} → {mapped}")
                    selected_class_name  = mapped
                    selected_color_label = chosen.lower()
            else:
                print("[Filter] 웹에서 전체 선택")
        except Exception as e:
            print(f"[Filter] 웹 선택 실패 → 콘솔 대체: {e}")

    # 2) 고정 설정 값
    if (selected_class_name is None) and (SELECTED_COLOR is not None):
        sc = SELECTED_COLOR.strip().lower()
        mapped = ALL_COLORS.get(sc)
        if mapped is None:
            print(f"[Filter] SELECTED_COLOR='{SELECTED_COLOR}' 무효 → 콘솔에서 선택")
            selected_class_name, selected_color_label = ask_color_and_map_to_class(ALL_COLORS)
        else:
            print(f"[Filter] 고정 선택 클래스: {mapped}")
            selected_class_name  = mapped
            selected_color_label = sc

    # 3) 콘솔 입력 대체
    if selected_class_name is None and (args.no_web or not _USE_WEB):
        selected_class_name, selected_color_label = ask_color_and_map_to_class(ALL_COLORS)

    # === 여기서 색상 라벨에 맞춰 CSV 파일명 생성 ===
    csv_label = _sanitize_label(selected_color_label) if selected_color_label else "all"
    CSV_GRIPS_PATH_dyn = f"grip_records_{csv_label}.csv"
    print(f"[Info] 경로 CSV: {CSV_GRIPS_PATH_dyn}")
    return selected_class_name, selected_color_label, CSV_GRIPS_PATH_dyn

def _open_cameras_and_model(size):
    capL_idx, capR_idx = CAM1_INDEX, CAM2_INDEX
    cap1, cap2 = open_cams(capL_idx, capR_idx, size)
    model = YOLO(str(MODEL_PATH))
    return cap1, cap2, model

def _initial_seg_merge(cap1, cap2, map1x, map1y, map2x, map2y, size, model, selected_class_name):
    print(f"[Init] First 10 frames: YOLO seg & merge ...")
    L_sets, R_sets = [], []
    for _ in range(2):
        cap1.read(); cap2.read()  # 워밍업

    for k in range(10):
        ok1, f1 = cap1.read(); ok2, f2 = cap2.read()
        if not (ok1 and ok2):
            cap1.release(); cap2.release()
            raise SystemExit("초기 프레임 캡쳐 실패")
        Lr_k = rectify(f1, map1x, map1y, size)
        Rr_k = rectify(f2, map2x, map2y, size)
        holdsL_k = extract_holds_with_indices(Lr_k, model, selected_class_name, THRESH_MASK, ROW_TOL_Y)
        holdsR_k = extract_holds_with_indices(Rr_k, model, selected_class_name, THRESH_MASK, ROW_TOL_Y)
        L_sets.append(holdsL_k); R_sets.append(holdsR_k)
        print(f"  - frame {k+1}/10: L={len(holdsL_k)}  R={len(holdsR_k)}")

    holdsL = assign_indices(merge_holds_by_center(L_sets, 18), ROW_TOL_Y)
    holdsR = assign_indices(merge_holds_by_center(R_sets, 18), ROW_TOL_Y)
    if not holdsL or not holdsR:
        cap1.release(); cap2.release()
        print("[Warn] 왼/오 프레임에서 홀드가 검출되지 않았습니다.")
        return None, None
    return holdsL, holdsR

def _build_common_ids(holdsL, holdsR):
    idxL = {h["hold_index"]: h for h in holdsL}
    idxR = {h["hold_index"]: h for h in holdsR}
    common_ids = sorted(set(idxL.keys()) & set(idxR.keys()))
    if not common_ids:
        print("[Warn] 좌/우 공통 hold_index가 없습니다.")
        return idxL, idxR, []
    print(f"[Info] 공통 홀드 개수: {len(common_ids)}")
    return idxL, idxR, common_ids

def _compute_matched_results(common_ids, idxL, idxR, P1, P2, L, O):
    matched_results = []
    for hid in common_ids:
        Lh = idxL[hid]; Rh = idxR[hid]
        X = triangulate_xy(P1, P2, Lh["center"], Rh["center"])
        d_left  = float(np.linalg.norm(X - L))
        d_line  = float(np.hypot(X[1], X[2]))
        yaw_deg, pitch_deg = yaw_pitch_from_X(X, O, Y_UP_IS_NEGATIVE)
        matched_results.append({
            "hid": hid, "color": Lh["color"],
            "X": X, "d_left": d_left, "d_line": d_line,
            "yaw_deg": yaw_deg, "pitch_deg": pitch_deg,
        })
    return matched_results

def _build_delta_tables(by_id, route_ids, O):
    next_id_map, delta_from_id, angle_deltas = {}, {}, []
    for i in range(len(route_ids)-1):
        a_id, b_id = route_ids[i], route_ids[i+1]
        if (a_id in by_id) and (b_id in by_id):
            a, b = by_id[a_id], by_id[b_id]
            dyaw   = wrap_deg(b["yaw_deg"]   - a["yaw_deg"])
            dpitch = wrap_deg(b["pitch_deg"] - a["pitch_deg"])
            v1 = a["X"] - O; v2 = b["X"] - O
            d3d = angle_between(v1, v2)
            angle_deltas.append((a_id, b_id, dyaw, dpitch, d3d))
            next_id_map[a_id]   = b_id
            delta_from_id[a_id] = (dyaw, dpitch)

    print("[ΔAngles] (CSV order):")
    for a_id, b_id, dyaw, dpitch, d3d in angle_deltas:
        print(f"  {a_id}->{b_id}: Δyaw={dyaw:+.2f}°, Δpitch={dpitch:+.2f}°, angle={d3d:.2f}°")
    return next_id_map, delta_from_id

def _triangulate_laser_3d(laser_px, P1, P2):
    if (not laser_px):
        return None
    Lp = laser_px.get("left_rect"); Rp = laser_px.get("right_rect")
    if (Lp is None) or (Rp is None):
        return None
    return triangulate_xy(P1, P2, Lp, Rp)

def _init_servo_and_point_first(ctl, args, current_target_id, by_id, X_laser, O,
                                cur_yaw, cur_pitch):
    try:
        ctl.set_angles(cur_pitch, cur_yaw)  # (pitch, yaw) 순서
        try:
            if args.laser_on:  # 존재하지 않을 수도 있으니 예외 무시
                ctl.laser_on()
        except:
            pass
    except Exception as e:
        print("[Init] Servo init error:", e)

    if (current_target_id is None) or (current_target_id not in by_id) or (X_laser is None):
        # 레이저 측정 실패 등 → 안전 폴백 (기존 절대각 또는 90/90 유지)
        print("[Init] 레이저 3D 또는 첫 타깃 없음 → 폴백 초기 조준 사용")
        if current_target_id is not None:
            mr0 = by_id[current_target_id]
            auto_yaw, auto_pitch = mr0["yaw_deg"], mr0["pitch_deg"]
            yaw_cmd0, pitch_cmd0 = to_servo_cmd(auto_yaw, auto_pitch)
            try:
                ctl.set_angles(pitch_cmd0, yaw_cmd0)
                cur_yaw, cur_pitch = yaw_cmd0, pitch_cmd0
            except Exception as e:
                print("[Init-Point Fallback] Servo move error:", e)
        return cur_yaw, cur_pitch

    # 레이저/홀드 기반 초기 조준
    X_hold = by_id[current_target_id]["X"]
    yaw_laser,  pitch_laser  = yaw_pitch_from_X(X_laser, O, Y_UP_IS_NEGATIVE)
    yaw_hold,   pitch_hold   = yaw_pitch_from_X(X_hold,  O, Y_UP_IS_NEGATIVE)
    d_yaw   = wrap_deg(yaw_hold  - yaw_laser) + CAL_YAW_OFFSET
    d_pitch = wrap_deg(pitch_hold - pitch_laser) + CAL_PITCH_OFFSET

    target_yaw   = BASE_YAW_DEG   + YAW_SIGN   * (YAW_SCALE   * d_yaw)
    target_pitch = BASE_PITCH_DEG + PITCH_SIGN * (PITCH_SCALE * d_pitch)
    target_yaw   = max(0.0, min(180.0, target_yaw))
    target_pitch = max(0.0, min(180.0, target_pitch))

    print(f"[Init-Target] laser yaw/pitch=({yaw_laser:.2f},{pitch_laser:.2f})°, "
          f"hold=({yaw_hold:.2f},{pitch_hold:.2f})°  "
          f"Δ=({d_yaw:+.2f},{d_pitch:+.2f})°  -> servo Y/P=({target_yaw:.2f},{target_pitch:.2f})")
    try:
        ctl.set_angles(target_pitch, target_yaw)  # (pitch, yaw) 순서
        cur_yaw, cur_pitch = target_yaw, target_pitch
    except Exception as e:
        print("[Init-Target] Servo move error:", e)
    return cur_yaw, cur_pitch

def _event_loop(cap1, cap2, size, map1x, map1y, map2x, map2y, SWAP_DISPLAY,
                laser_px, holdsL, holdsR, matched_results, by_id,
                next_id_map, delta_from_id, args):
    W, H = size
    pose = PoseTracker(min_detection_confidence=0.5, model_complexity=1)
    touch = TouchCounter(threshold_frames=TOUCH_THRESHOLD, cooldown_sec=ADV_COOLDOWN)

    filled_ids = set()   # 성공 처리된(채워진) 홀드 ID
    blocked_state = {}   # (part, hold_id)별 차폐 상태

    out = None
    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(OUT_PATH, fourcc, OUT_FPS, (W*2, H))

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    t_prev = time.time()
    last_advanced_time = 0.0

    # 외부에서 현재 타깃, 서보 각도는 관리하므로 반환 값으로 전달
    return pose, touch, filled_ids, blocked_state, out, t_prev, last_advanced_time

def _run_frame_loop(cap1, cap2, map1x, map1y, map2x, map2y, size,
                    SWAP_DISPLAY, laser_px,
                    holdsL, holdsR, matched_results,
                    by_id, next_id_map, delta_from_id,
                    auto_advance_enabled,
                    pose, touch, filled_ids, blocked_state,
                    out, t_prev, last_advanced_time,
                    current_target_id, cur_yaw, cur_pitch, ctl):
    W, H = size
    try:
        while True:
            ok1, f1 = cap1.read(); ok2, f2 = cap2.read()
            if not (ok1 and ok2):
                print("[Warn] 프레임 캡쳐 실패"); break

            Lr = rectify(f1, map1x, map1y, size)
            Rr = rectify(f2, map2x, map2y, size)

            # (옵션) 레이저 점 시각 확인
            if laser_px:
                if laser_px["left_rect"] is not None:
                    lx, ly = laser_px["left_rect"]
                    cv2.circle(Lr, (lx, ly), 8, (0,0,255), 2, cv2.LINE_AA)
                if laser_px["right_rect"] is not None:
                    rx, ry = laser_px["right_rect"]
                    cv2.circle(Rr, (rx, ry), 8, (0,0,255), 2, cv2.LINE_AA)

            vis = np.hstack([Rr, Lr]) if SWAP_DISPLAY else np.hstack([Lr, Rr])

            # 검출 결과 그리기(성공 홀드는 반투명 칠하기)
            for side, holds in (("L", holdsL), ("R", holdsR)):
                xoff = xoff_for(side, W, SWAP_DISPLAY)
                for h in holds:
                    cnt_shifted = h["contour"] + np.array([[[xoff, 0]]], dtype=h["contour"].dtype)

                    if h["hold_index"] in filled_ids:
                        overlay = vis.copy()
                        cv2.drawContours(overlay, [cnt_shifted], -1, h["color"], thickness=-1)
                        vis = cv2.addWeighted(overlay, 0.45, vis, 0.55, 0)

                    cv2.drawContours(vis, [cnt_shifted], -1, h["color"], 2)
                    cx, cy = h["center"]
                    cv2.circle(vis, (cx+xoff, cy), 4, (255,255,255), -1)
                    tag = f"ID:{h['hold_index']}"
                    if (current_target_id is not None) and (h["hold_index"] == current_target_id):
                        tag = "[TARGET] " + tag
                    cv2.putText(vis, tag, (cx+xoff-10, cy+26),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
                    cv2.putText(vis, tag, (cx+xoff-10, cy+26),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, h["color"], 2, cv2.LINE_AA)

            # --- 디버그 3D 좌표/깊이 표시 ---
            y_info = 60
            for mr in matched_results:
                X = mr["X"]
                depth = X[2]
                txt3d = (f"ID{mr['hid']} : X=({X[0]:.1f}, {X[1]:.1f}, {X[2]:.1f}) mm "
                         f" | depth(Z)={depth:.1f} mm")
                cv2.putText(vis, txt3d, (20, y_info),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA)
                cv2.putText(vis, txt3d, (20, y_info),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
                y_info += 18

            # NEXT 텍스트 및 현재 각도 표시
            y0 = 28
            if current_target_id in by_id:
                if current_target_id in delta_from_id:
                    dyaw, dpitch = delta_from_id[current_target_id]
                    nxt = next_id_map[current_target_id]
                    txt = (f"[NEXT] ID{current_target_id}→ID{nxt}  "
                           f"Δyaw={dyaw:+.1f}°, Δpitch={dpitch:+.1f}°  "
                           f"[now yaw={cur_yaw:.1f}°, pitch={cur_pitch:.1f}°]")
                else:
                    mr = by_id[current_target_id]
                    txt = (f"[LAST] ID{mr['hid']}  yaw={mr['yaw_deg']:.1f}°, pitch={mr['pitch_deg']:.1f}°  "
                           f"[now yaw={cur_yaw:.1f}°, pitch={cur_pitch:.1f}°]")
                cv2.putText(vis, txt, (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(vis, txt, (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 1, cv2.LINE_AA)

            # === MediaPipe 포즈 추정 & 표시 ===
            coords = pose.process(Lr)
            left_xoff = xoff_for("L", W, SWAP_DISPLAY)
            draw_pose_points(vis, coords, offset_x=left_xoff)

            # === (핵심) 판정 스냅샷 ID 사용 & 프레임당 1회만 타깃 변경 ===
            if coords and (current_target_id in {h["hold_index"] for h in holdsL}):
                tid = current_target_id
                hold = {h["hold_index"]: h for h in holdsL}[tid]

                _ = touch.check(hold["contour"], coords, tid, now=time.time())  # triggered, parts (로직 유지)
                advanced_this_frame = False

                for name, (x, y) in coords.items():
                    key = (name, tid)
                    inside = cv2.pointPolygonTest(hold["contour"], (x, y), False) >= 0

                    if inside:
                        # 성공 파트
                        if name in pose.success_parts:
                            filled_ids.add(tid)
                            now_t = time.time()
                            if (auto_advance_enabled and tid in delta_from_id
                                and (now_t - last_advanced_time) > ADV_COOLDOWN
                                and not advanced_this_frame):

                                dyaw, dpitch = delta_from_id[tid]
                                target_yaw   = cur_yaw   - dyaw
                                target_pitch = cur_pitch + dpitch
                                send_servo_angles(ctl, target_yaw, target_pitch)
                                cur_yaw, cur_pitch = target_yaw, target_pitch

                                current_target_id  = next_id_map[tid]
                                last_advanced_time = now_t
                                advanced_this_frame = True
                                break

                        # 차폐 파트
                        elif name in pose.blocking_parts:
                            if not blocked_state.get(key, False):
                                blocked_state[key] = True
                    else:
                        blocked_state[key] = False

            # FPS
            t_now = time.time()
            fps = 1.0 / max(t_now - (t_prev), 1e-6); t_prev = t_now
            cv2.putText(vis, f"FPS: {fps:.1f} (Auto={'ON' if auto_advance_enabled else 'OFF'})",
                        (10, H-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(vis, f"FPS: {fps:.1f} (Auto={'ON' if auto_advance_enabled else 'OFF'})",
                        (10, H-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1, cv2.LINE_AA)

            imshow_scaled(WINDOW_NAME, vis, None)
            if SAVE_VIDEO:
                if out is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(OUT_PATH, fourcc, OUT_FPS, (W*2, H))
                out.write(vis)

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            elif k == ord('n') and (current_target_id in delta_from_id):
                # (수동) 다음 이동
                filled_ids.add(current_target_id)
                dyaw, dpitch = delta_from_id[current_target_id]
                target_yaw   = cur_yaw   - dyaw
                target_pitch = cur_pitch + dpitch
                send_servo_angles(ctl, target_yaw, target_pitch)
                cur_yaw, cur_pitch = target_yaw, target_pitch
                current_target_id  = next_id_map[current_target_id]
                print(f"[Manual Next] moved with Δ (dyaw={dyaw:+.2f}, dpitch={dpitch:+.2f})")
                continue

    finally:
        cap1.release(); cap2.release()
        if SAVE_VIDEO and out is not None:
            out.release(); print(f"[Info] 저장 완료: {OUT_PATH}")
        cv2.destroyAllWindows()
        try: pose.close()
        except: pass
        try: ctl.close()
        except: pass

    # 상태 업데이트 반환(필요시 확장용)
    return current_target_id, cur_yaw, cur_pitch

# ---------- 메인 ----------
def main():
    args = _parse_args()

    # 경로 검증
    _verify_paths()

    # 스테레오 로드
    map1x, map1y, map2x, map2y, P1, P2, size, (W, H) = _load_stereo_and_log()

    # 레이저 좌표 먼저 측정 (find_laser) → 보정좌표로 변환
    laser_px = _capture_laser_and_rectify(args, W, H)

    # 레이저 원점 O (LEFT 기준)
    L, O = _compute_laser_origin_left()
    print(L, O)

    # 색상 필터 선택 (A_web → 고정 → 콘솔)
    selected_class_name, selected_color_label, CSV_GRIPS_PATH_dyn = _choose_color(args)

    # 카메라 & 모델
    cap1, cap2, model = _open_cameras_and_model(size)

    # 초기 10프레임: YOLO seg & merge
    holdsL, holdsR = _initial_seg_merge(cap1, cap2, map1x, map1y, map2x, map2y, size,
                                        model, selected_class_name)
    if not holdsL or not holdsR:
        return

    # 공통 ID
    idxL, idxR, common_ids = _build_common_ids(holdsL, holdsR)
    if not common_ids:
        return

    # 3D/각도 계산
    matched_results = _compute_matched_results(common_ids, idxL, idxR, P1, P2, L, O)

    # Δ 테이블 (CSV 순서 기반)
    by_id  = {mr["hid"]: mr for mr in matched_results}
    route_ids = load_route_ids_from_csv(CSV_GRIPS_PATH_dyn)
    if not route_ids:
        route_ids = sorted(by_id.keys())
    next_id_map, delta_from_id = _build_delta_tables(by_id, route_ids, O)

    # Servo 초기화 & '레이저→첫 홀드' Δ각 기반 초기 조준
    ctl = DualServoController(args.port, args.baud) if HAS_SERVO else DualServoController()

    # 1) 첫 타깃 ID 결정 (CSV 우선, 없으면 최솟값)
    if route_ids:
        current_target_id = route_ids[0]
    else:
        current_target_id = min(by_id.keys()) if by_id else None

    # 2) 레이저 3D
    X_laser = _triangulate_laser_3d(laser_px, P1, P2)

    # 3) 서보는 중립 90/90에서 시작(규약), Δ각을 적용해 첫 홀드로
    cur_yaw, cur_pitch = BASE_YAW_DEG, BASE_PITCH_DEG
    cur_yaw, cur_pitch = _init_servo_and_point_first(ctl, args, current_target_id, by_id,
                                                     X_laser, O, cur_yaw, cur_pitch)

    # 루프 준비 객체 생성 (상태는 아래 프레임 루프에서 유지/갱신)
    pose, touch, filled_ids, blocked_state, out, t_prev, last_advanced_time = \
        _event_loop(cap1, cap2, size, map1x, map1y, map2x, map2y, SWAP_DISPLAY,
                    laser_px, holdsL, holdsR, matched_results, by_id,
                    next_id_map, delta_from_id, args)

    # 본 루프 실행
    _ = _run_frame_loop(cap1, cap2, map1x, map1y, map2x, map2y, size,
                        SWAP_DISPLAY, laser_px,
                        holdsL, holdsR, matched_results,
                        by_id, next_id_map, delta_from_id,
                        (not args.no_auto_advance),
                        pose, touch, filled_ids, blocked_state,
                        out, t_prev, last_advanced_time,
                        current_target_id, cur_yaw, cur_pitch, ctl)

if __name__ == "__main__":
    main()
