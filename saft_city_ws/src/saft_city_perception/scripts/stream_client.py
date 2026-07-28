#!/usr/bin/env python3
"""主机端 — 接收仿真画面并显示"""
import socket, struct, pickle, cv2, sys
import numpy as np

HOST = '192.168.1.100'  # 改成虚拟机的IP
PORT = 8888

def main():
    if len(sys.argv) > 1:
        ip = sys.argv[1]
    else:
        ip = HOST

    print(f"连接 {ip}:{PORT}...")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(10)
    try:
        s.connect((ip, PORT))
        print("✅ 已连接！按 ESC 退出")
    except:
        print("❌ 连接失败，请检查IP")
        return

    while True:
        try:
            # 收图片大小
            size_data = s.recv(4)
            if len(size_data) < 4: break
            size = struct.unpack('>I', size_data)[0]

            # 收图片数据
            buf = b''
            while len(buf) < size:
                chunk = s.recv(min(size - len(buf), 65536))
                if not chunk: break
                buf += chunk
            if len(buf) < size: break

            # 显示图片
            img = pickle.loads(buf)
            cv2.imshow('平安城市 - 相机画面', img)
            key = cv2.waitKey(30)
            if key == 27:  # ESC
                break
        except:
            break

    cv2.destroyAllWindows()
    s.close()
    print("已断开")

if __name__ == '__main__':
    main()
