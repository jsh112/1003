# servo_control.py
import argparse
import time
import sys
import serial

READY_WAIT_SEC = 2.0   # 아두이노 자동리셋 대기

class DualServoController:
    def __init__(self, port="COM15", baud=115200, timeout=1.0):
        self.ser = serial.Serial(port, baudrate=baud, timeout=timeout)
        time.sleep(READY_WAIT_SEC)
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
        # 웜업: 상태 한 번 읽기(없어도 무방)
        try:
            self.query()
        except Exception:
            pass

    # ---------------- low-level ----------------
    def _send(self, line: str) -> str:
        """한 줄 명령을 보내고, 한 줄 응답을 문자열로 반환."""
        if not line.endswith("\n"):
            line += "\n"
        self.ser.write(line.encode("utf-8"))
        self.ser.flush()
        resp = self.ser.readline().decode("utf-8", errors="ignore").strip()
        return resp

    # ---------------- high-level ----------------
    def set_angles(self, pitch_deg: float, yaw_deg: float) -> str:
        """두 축 절대각 명령: 아두이노 'S p y'"""
        cmd = f"S {float(pitch_deg):.2f} {float(yaw_deg):.2f}"
        resp = self._send(cmd)
        return resp  # 예: "OK S 96.80 88.50"

    def set_pitch(self, pitch_deg: float) -> str:
        return self._send(f"P {float(pitch_deg):.2f}")

    def set_yaw(self, yaw_deg: float) -> str:
        return self._send(f"Y {float(yaw_deg):.2f}")

    def center(self) -> str:
        return self._send("C")

    def query(self):
        """상태 질의: 'STATE pitch yaw laser' 형태를 파싱해 dict 반환."""
        resp = self._send("Q")
        # 예: "STATE 96.80 88.50 1"
        out = {"raw": resp, "pitch": None, "yaw": None, "laser": None}
        try:
            tok = resp.split()
            if tok[0].upper() == "STATE":
                out["pitch"] = float(tok[1])
                out["yaw"]   = float(tok[2])
                out["laser"] = int(tok[3]) if len(tok) > 3 else None
        except Exception:
            pass
        return out

    # ----------- Laser (NEW) -----------
    def laser_on(self) -> str:
        """레이저 ON: 'L true'"""
        return self._send("L true")

    def laser_off(self) -> str:
        """레이저 OFF: 'L false'"""
        return self._send("L false")

    # -----------------------------------
    def close(self):
        try:
            self.ser.close()
        except Exception:
            pass


# ---------------- CLI test ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM15")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--center", action="store_true")
    ap.add_argument("--pitch", type=float, help="absolute pitch (deg)")
    ap.add_argument("--yaw",   type=float, help="absolute yaw (deg)")
    ap.add_argument("--laser-on",  action="store_true")
    ap.add_argument("--laser-off", action="store_true")
    ap.add_argument("--query", action="store_true")
    args = ap.parse_args()

    ctl = DualServoController(args.port, args.baud)
    try:
        if args.center:
            print(ctl.center())
        if args.pitch is not None and args.yaw is not None:
            print(ctl.set_angles(args.pitch, args.yaw))
        elif args.pitch is not None:
            print(ctl.set_pitch(args.pitch))
        elif args.yaw is not None:
            print(ctl.set_yaw(args.yaw))

        if args.laser_on:
            print(ctl.laser_on())
        if args.laser_off:
            print(ctl.laser_off())

        if args.query or (not any([args.center, args.pitch is not None, args.yaw is not None, args.laser_on, args.laser_off])):
            print(ctl.query())
    finally:
        ctl.close()


if __name__ == "__main__":
    main()
