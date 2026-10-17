"""
AttenMind API Server
====================
Wraps the exact same model pipeline from attenmind_app_2.py into a
REST API so the website can call it.

Usage:
    pip install flask flask-cors tensorflow torch torchvision dlib opencv-python-headless Pillow numpy
    python attenmind_api.py

Endpoints:
    POST /analyze   — accepts multipart image or base64 JSON, returns per-face JSON
    GET  /health    — returns model load status
"""

import os, io, glob, base64, traceback
import cv2, numpy as np, dlib, torch, torch.nn as nn, tensorflow as tf
from PIL import Image
from torchvision import models
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── Configuration ─────────────────────────────────────────────────────────────
SCRIPT_DIR          = os.path.dirname(os.path.abspath(__file__))
DRIVE_FOLDER        = r'c:\Users\Rahaf\Downloads'   # original path — fallback

EYE_MODEL_PATH      = os.path.join(SCRIPT_DIR, 'new_eye_classifier_model (1).keras')
YAWN_MODEL_PATH     = os.path.join(SCRIPT_DIR, 'best_model_final.h5')
HEADPOSE_MODEL_PATH = os.path.join(SCRIPT_DIR, 'headpose_attenmind (1).pt')
DLIB_PREDICTOR_PATH = os.path.join(SCRIPT_DIR, 'shape_predictor_68_face_landmarks (1).dat')

app = Flask(__name__)
CORS(app)

eye_model = yawn_model = hp_model = face_detector = dlib_predictor = None
device = 'cpu'
_load_error = None
_load_warnings = []


def build_candidates(primary, patterns):
    roots, seen, result = [], set(), []
    for root in [SCRIPT_DIR, DRIVE_FOLDER, os.getcwd()]:
        if root and os.path.isdir(root) and root not in roots:
            roots.append(root)
    def add(p):
        n = os.path.normcase(os.path.abspath(p))
        if n not in seen: seen.add(n); result.append(p)
    if os.path.exists(primary): add(primary)
    for root in roots:
        for pat in patterns:
            for p in glob.glob(os.path.join(root, '**', pat), recursive=True): add(p)
    return result


def load_all_models():
    global eye_model, yawn_model, hp_model, face_detector, dlib_predictor
    global device, _load_error, _load_warnings
    try:
        # Eye
        for p in build_candidates(EYE_MODEL_PATH, ['*eye*.keras','*eye*.h5']):
            try:
                eye_model = tf.keras.models.load_model(p)
                if p != EYE_MODEL_PATH: _load_warnings.append(f'Eye fallback: {os.path.basename(p)}')
                print(f'[OK] Eye model: {p}'); break
            except Exception as e: print(f'[SKIP] {p}: {e}')
        if eye_model is None:
            raise RuntimeError('Eye model not found. Place new_eye_classifier_model.keras next to this script.')

        # Yawn
        for p in build_candidates(YAWN_MODEL_PATH, ['*yawn*.h5','*final*.h5','*mouth*.h5']):
            try:
                yawn_model = tf.keras.models.load_model(p)
                if p != YAWN_MODEL_PATH: _load_warnings.append(f'Yawn fallback: {os.path.basename(p)}')
                print(f'[OK] Yawn model: {p}'); break
            except Exception as e: print(f'[SKIP] {p}: {e}')
        if yawn_model is None:
            raise RuntimeError('Yawn model not found. Place best_model_final.h5 next to this script.')

        # Head-pose
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        checkpoint = None
        for p in build_candidates(HEADPOSE_MODEL_PATH, ['*headpose*.pt','*head_pose*.pt']):
            try:
                checkpoint = torch.load(p, map_location=device)
                if p != HEADPOSE_MODEL_PATH: _load_warnings.append(f'Head-pose fallback: {os.path.basename(p)}')
                print(f'[OK] Head-pose: {p}'); break
            except Exception as e: print(f'[SKIP] {p}: {e}')

        class HeadPoseModel(nn.Module):
            def __init__(self, yaw_bins=51, pitch_bins=41):
                super().__init__()
                self.backbone    = models.resnet18(weights=None)
                self.backbone.fc = nn.Identity()
                self.yaw_head    = nn.Linear(512, yaw_bins)
                self.pitch_head  = nn.Linear(512, pitch_bins)
            def forward(self, x):
                x = self.backbone(x)
                return self.pitch_head(x), self.yaw_head(x)

        if checkpoint is None:
            hp_model = None
            _load_warnings.append('Head-pose model missing — geometry fallback only.')
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            sd = checkpoint['state_dict']
            hp_model = HeadPoseModel(sd['yaw_head.weight'].shape[0], sd['pitch_head.weight'].shape[0]).to(device)
            hp_model.load_state_dict(sd, strict=False); hp_model.eval()
        else:
            hp_model = checkpoint.to(device); hp_model.eval()

        # dlib
        face_detector = dlib.get_frontal_face_detector()
        for p in build_candidates(DLIB_PREDICTOR_PATH, ['shape_predictor_68*.dat']):
            if os.path.exists(p):
                dlib_predictor = dlib.shape_predictor(p)
                print(f'[OK] dlib predictor: {p}'); break
        if dlib_predictor is None:
            raise RuntimeError('shape_predictor_68_face_landmarks.dat not found. Download from dlib.net.')

        print('[OK] All models ready.')
    except Exception as e:
        _load_error = str(e); print(f'[ERROR] {e}'); traceback.print_exc()


# ── Pipeline (exact logic from attenmind_app_2.py) ────────────────────────────

def detect_all_faces(img_rgb):
    gray  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    rects = face_detector(gray, 1)
    faces = [(int(r.left()),int(r.top()),int(r.right()-r.left()),int(r.bottom()-r.top())) for r in rects]
    faces.sort(key=lambda b: b[0]+b[2]/2.0)
    return faces

def detect_regions_with_dlib(img_rgb, face_box):
    x,y,fw,fh = face_box
    gray  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    rect  = dlib.rectangle(int(x),int(y),int(x+fw),int(y+fh))
    shape = dlib_predictor(gray, rect)
    pts   = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)], dtype=np.int32)
    def box_for(idx, pad=0.25):
        sub=pts[idx]; x1,y1=sub.min(axis=0); x2,y2=sub.max(axis=0)
        dx=int((x2-x1)*pad); dy=int((y2-y1)*pad)
        return (x1-dx,y1-dy,x2+dx,y2+dy)
    return box_for(list(range(36,42)),0.35), box_for(list(range(42,48)),0.35), box_for(list(range(48,68)),0.2), pts

def crop_bbox(img, bbox):
    h,w=img.shape[:2]; x1,y1,x2,y2=bbox
    x1,y1=max(0,int(x1)),max(0,int(y1)); x2,y2=min(w,int(x2)),min(h,int(y2))
    if x2<=x1 or y2<=y1: return np.zeros((32,32,3),dtype=np.uint8)
    return img[y1:y2,x1:x2]

def crop_for_headpose(img, face_box, pad=0.45):
    x,y,w,h=face_box; cx,cy=x+w/2.0,y+h/2.0; side=float(max(w,h))*(1+2*pad)
    x1,y1=max(0,int(round(cx-side/2))),max(0,int(round(cy-side/2)))
    x2,y2=min(img.shape[1],int(round(cx+side/2))),min(img.shape[0],int(round(cy+side/2)))
    if x2<=x1 or y2<=y1: return img
    return img[y1:y2,x1:x2]

def preprocess_keras(img, size, ch=3):
    img=cv2.resize(img,(size[1],size[0]))
    if ch==1: img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY); img=np.expand_dims(img,-1)
    elif img.ndim==2: img=cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    return np.expand_dims(img.astype('float32')/255.0,0)

def preprocess_torch(img):
    img=cv2.resize(img,(224,224)).astype('float32')/255.0
    return torch.tensor(np.transpose(img,(2,0,1))).unsqueeze(0).float().to(device)

def predict_eye(img):
    ch=int(eye_model.input_shape[-1] or 3)
    pred=np.asarray(eye_model.predict(preprocess_keras(img,eye_model.input_shape[1:3],ch),verbose=0)[0]).reshape(-1)
    return ('open' if np.argmax(pred)==1 else 'closed'), float(pred[1] if pred.size>1 else pred[0])

def compute_ear(pts):
    p=np.asarray(pts,dtype=np.float32); p1,p2,p3,p4,p5,p6=p
    h=np.linalg.norm(p1-p4)
    return float((np.linalg.norm(p2-p6)+np.linalg.norm(p3-p5))/(2.0*h)) if h>1e-6 else 0.0

def predict_yawn(img):
    ch=int(yawn_model.input_shape[-1] or 3)
    pred=np.asarray(yawn_model.predict(preprocess_keras(img,yawn_model.input_shape[1:3],ch),verbose=0)[0]).reshape(-1)
    return ('yawn' if np.argmax(pred)==1 else 'no_yawn'), float(pred[1] if pred.size>1 else pred[0])

def compute_mar(pts):
    p=np.asarray(pts,dtype=np.float32); h=np.linalg.norm(p[0]-p[6])
    return float((np.linalg.norm(p[2]-p[10])+np.linalg.norm(p[3]-p[9]))/(2.0*h)) if h>1e-6 else 0.0

def predict_headpose(img):
    if hp_model is None: return 'forward',0.0,0.0
    pitch,yaw=hp_model(preprocess_torch(img))
    def to_angle(v):
        arr=np.asarray(v.cpu().detach()).reshape(-1)
        return float(arr[0]) if arr.size==1 else float(np.argmax(arr)-(arr.size-1)/2.0)
    yv,pv=to_angle(yaw),to_angle(pitch)
    return ('distracted' if abs(yv)>20 or abs(pv)>15 else 'forward'),yv,pv

def clamp01(v): return float(max(0.0,min(1.0,v)))
def ramp(v,lo,hi):
    if hi<=lo: return 1.0 if v>=hi else 0.0
    return clamp01((v-lo)/(hi-lo))

def compute_score(ec,yc,hc):
    s=int(round(40*ec+30*yc+30*hc))
    if s>=80:   return s,'Focused','focused'
    elif s>=50: return s,'Moderate','moderate'
    else:       return s,'Distracted','distracted'

def analyze_single_face(img_rgb, face_box):
    leb,reb,mbox,lm68 = detect_regions_with_dlib(img_rgb,face_box)
    le,re,mo = crop_bbox(img_rgb,leb), crop_bbox(img_rgb,reb), crop_bbox(img_rgb,mbox)

    _,lp = predict_eye(le); _,rp = predict_eye(re)
    model_eye_open = (lp+rp)/2.0
    avg_ear = (compute_ear(lm68[36:42])+compute_ear(lm68[42:48]))/2.0
    ear_open,model_open = avg_ear>=0.22, model_eye_open>=0.55
    if ear_open and model_open: eye='open'
    elif (not ear_open) and (not model_open): eye='closed'
    else: eye='open' if ear_open else 'closed'

    _,yawn_prob = predict_yawn(mo)
    mar = compute_mar(lm68[48:60])
    mar_yawn,model_yawn = mar>=0.33, yawn_prob>=0.55
    if mar_yawn and model_yawn: yawn='yawn'
    elif (not mar_yawn) and (not model_yawn): yawn='no_yawn'
    else: yawn='yawn' if mar_yawn else 'no_yawn'

    head_label,yaw_angle,pitch_angle = predict_headpose(crop_for_headpose(img_rgb,face_box))
    nose        = lm68[30].astype(np.float32)
    left_eye_c  = lm68[36:42].mean(axis=0).astype(np.float32)
    right_eye_c = lm68[42:48].mean(axis=0).astype(np.float32)
    mouth_c     = lm68[48:68].mean(axis=0).astype(np.float32)
    eye_center  = (left_eye_c+right_eye_c)/2.0
    eye_span    = np.linalg.norm(right_eye_c-left_eye_c)
    nose_dx=nose_dy=0.0; geom_fwd=True
    if eye_span>1e-6:
        nose_dx=abs(float((nose[0]-eye_center[0])/eye_span))
        nose_dy=abs(float((nose[1]-(eye_center[1]+mouth_c[1])/2.0)/eye_span))
        geom_fwd=(nose_dx<0.20) and (nose_dy<0.35)
    model_fwd=(head_label=='forward')
    if geom_fwd and model_fwd: head='forward'
    elif (not geom_fwd) and (not model_fwd): head='distracted'
    else: head='forward' if geom_fwd else 'distracted'

    eye_conf  = 0.6*ramp(avg_ear,0.18,0.30) + 0.4*clamp01(model_eye_open)
    yawn_conf = 0.55*(1.0-ramp(mar,0.25,0.40)) + 0.45*clamp01(1.0-yawn_prob)
    pose_conf = 0.6*(1.0-ramp(abs(yaw_angle),10.0,35.0)) + 0.4*(1.0-ramp(abs(pitch_angle),8.0,25.0))
    geom_head_conf = 1.0-ramp((nose_dx/0.22)+(nose_dy/0.36),0.9,1.5)
    head_conf = 0.7*pose_conf + 0.3*geom_head_conf

    s,status,css = compute_score(eye_conf,yawn_conf,head_conf)
    return {
        'score':s,'status':status,'css':css,
        'eye':eye,'yawn':yawn,'head':head,
        'eye_conf':round(eye_conf,3),'yawn_conf':round(yawn_conf,3),'head_conf':round(head_conf,3),
        'avg_ear':round(float(avg_ear),3),'mar':round(float(mar),3),
        'yaw_angle':round(float(yaw_angle),2),'pitch_angle':round(float(pitch_angle),2),
        'face_box':list(face_box),
    }

def ndarray_to_b64(arr):
    if arr is None or arr.size==0: return None
    buf=io.BytesIO(); Image.fromarray(arr.astype(np.uint8)).save(buf,format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    if _load_error:
        return jsonify({'ok':False,'error':_load_error,'warnings':_load_warnings}), 503
    return jsonify({'ok':True,'warnings':_load_warnings,'device':device})

@app.route('/analyze', methods=['POST'])
def analyze():
    if _load_error:
        return jsonify({'ok':False,'error':'Models not loaded: '+_load_error}), 503
    try:
        if 'image' in request.files:
            file_bytes = request.files['image'].read()
        elif request.is_json and 'image' in request.json:
            b64 = request.json['image']
            if ',' in b64: b64 = b64.split(',',1)[1]
            file_bytes = base64.b64decode(b64)
        else:
            return jsonify({'ok':False,'error':'No image provided. Send multipart "image" or JSON {"image":"<base64>"}'}), 400

        nparr   = np.frombuffer(file_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return jsonify({'ok':False,'error':'Could not decode image.'}), 400
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        faces = detect_all_faces(img_rgb)
        if not faces:
            return jsonify({'ok':True,'faces_detected':0,'faces':[],'class_average':None,
                            'message':'No face detected. Use a clear frontal image.'})

        annotated = img_rgb.copy()
        results   = []
        for idx,face_box in enumerate(faces, start=1):
            row = analyze_single_face(img_rgb, face_box)
            row['student_index'] = idx
            results.append(row)
            x,y,fw,fh = face_box
            cv2.rectangle(annotated,(x,y),(x+fw,y+fh),(214,153,232),2)
            cv2.putText(annotated,f'S{idx}: {row["score"]}',(x,max(0,y-8)),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(252,191,73),1)

        scores = [r['score'] for r in results]
        avg_s  = int(round(float(np.mean(scores))))
        if avg_s>=80:   avg_status,avg_css='Focused','focused'
        elif avg_s>=50: avg_status,avg_css='Moderate','moderate'
        else:           avg_status,avg_css='Distracted','distracted'

        return jsonify({
            'ok':True,
            'faces_detected':len(results),
            'faces':results,
            'class_average':{'score':avg_s,'status':avg_status,'css':avg_css},
            'annotated_image': ndarray_to_b64(annotated),
            'warnings':_load_warnings,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'ok':False,'error':str(e)}), 500

if __name__ == '__main__':
    print('='*60)
    print('  AttenMind API Server  —  loading models…')
    print('='*60)
    load_all_models()
    if _load_error:
        print(f'  ⚠  {_load_error}')
    else:
        print('  ✓  All models ready → http://localhost:5050')
    print('='*60)
    app.run(host='0.0.0.0', port=5050, debug=False)
