# calcOPticalFlowFarneback 추적 (track_optical_farneback.py)
# https://github.com/BaekKyunShin/OpenCV_Project_Python/tree/master/08.match_track
import cv2, numpy as np

# 플로우 결과 그리기 ---①
def drawFlow(lines, img,flow,step=16):
    h,w = img.shape[:2]
    # 16픽셀 간격의 그리드 인덱스 구하기 ---②
    idx_y,idx_x = np.mgrid[step/2:h:step,step/2:w:step].astype(np.int)
    indices =  np.stack( (idx_x,idx_y), axis =-1).reshape(-1,2)
    # 추적선 그릴 이미지를 프레임 크기에 맞게 생성
    optical_flow_array = []
    for x,y in indices:   # 인덱스 순회
        # 각 그리드 인덱스 위치에 점 그리기 ---③
        cv2.circle(img, (x,y), 1, (0,255,0), -1)
        # 각 그리드 인덱스에 해당하는 플로우 결과 값 (이동 거리)  ---④
        dx,dy = flow[y, x].astype(np.int)
        distance = (dx**2 + dy**2)**0.5
        # 각 그리드 인덱스 위치에서 이동한 거리 만큼 선 그리기 ---⑤
        cv2.line(img, (x,y), (x+dx, y+dy), (0,255, 0),2, cv2.LINE_AA )



roots = 'ball_t1.mp4'
output_root = 'Gunner.mp4'

prev = None # 이전 프레임 저장 변수

cap = cv2.VideoCapture(roots)
fps = cap.get(cv2.CAP_PROP_FPS) # 프레임 수 구하기
delay = int(100/fps)
frame_array = []
while cap.isOpened():
    ret,frame = cap.read()
    if not ret: 
        break
# frame = cv2.resize(frame, dsize=(640, 480), interpolation=cv2.INTER_AREA)
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    img_draw = frame.copy()
    height,width,layers = img_draw.shape
    size = (width,height)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
  # 최초 프레임 경우 
    if prev is None: 
        prev = gray # 첫 이전 프레임 --- ⑥
        # 추적선 그릴 이미지를 프레임 크기에 맞게 생성
        lines = np.zeros_like(frame)
    else:
        # 이전, 이후 프레임으로 옵티컬 플로우 계산 ---⑦
        flow = cv2.calcOpticalFlowFarneback(prev,gray,None,\
                    0.5,3,15,3,5,1.1,cv2.OPTFLOW_FARNEBACK_GAUSSIAN) 
        # 계산 결과 그리기, 선언한 함수 호출 ---⑧
        drawFlow(lines,frame,flow)
        # 다음 프레임을 위해 이월 ---⑨
        prev = gray
        
    frame_array.append(frame)
    cv2.imshow('OpticalFlow-Farneback', frame)
    if cv2.waitKey(delay) == 27:
        break

out = cv2.VideoWriter(output_root, cv2.VideoWriter_fourcc(*'DIVX'),int(fps/3),size)
for frame in frame_array:
    out.write(frame)
out.release()
cap.release()
cv2.destroyAllWindows()