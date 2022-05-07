import cv2
import numpy as np
import onnxruntime as rt
import math

class faceDetectorAndAlignment:
    def __init__(self, modelFile, processScale=1):
        sessOptions = rt.SessionOptions()
        sessOptions.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL 

        self.detector = rt.InferenceSession(modelFile, sessOptions)
        self.transDst = np.array([
                [30.2946, 51.6963],
                [65.5318, 51.5014],
                [48.0252, 71.7366],
                [33.5493, 92.3655],
                [62.7299, 92.2041]], dtype=np.float32)
        self.transDst[:, 0] += 8.0
        
        self.processScale = processScale
        self.stride = 4

    def calcImageScale(self, h, w):
        hNew, wNew = int(np.ceil(h / 32) * 32), int(np.ceil(w / 32) * 32)
        ratioH, ratioW = hNew / h, wNew / w
        return (hNew, wNew), (ratioH, ratioW)

    def nms(self, dets, thresh):
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep

    def exp(self, item):
        for idx in range(len(item)):
            if abs(item[idx]) < 1:
                item[idx] *= math.e
            elif item[idx] > 0:
                item[idx] = math.exp(item[idx])
            else:
                item[idx] = -math.exp(-item[idx])
            
        return item
        
    
    def map2Box(self, cx, cy, scores, box, landmark, ratioW, ratioH, threshold=0.3):
        xs, ys = cx.astype(np.int32).squeeze(), cy.astype(np.int32).squeeze()
        scores, box, landmark = scores.squeeze(), box.squeeze(), landmark.squeeze()
        
        positiveIdx = np.where(scores >=threshold)[0]
        faceBoxes = np.empty((len(positiveIdx), 5), dtype=np.float32)
        faceLandmarks = np.empty((len(positiveIdx), 10), dtype=np.float32)
        
        for idx, positivetIdx in enumerate(positiveIdx):
            cx, cy, score = xs[positivetIdx], ys[positivetIdx], scores[positivetIdx]
            x, y ,r, b = box[:, cy, cx]
            faceBoxes[idx, 0:4] = (np.array([cx - x, cy - y, cx + r, cy + b]) * self.stride) / [ratioW, ratioH, ratioW, ratioH]
            faceBoxes[idx, 4] = score

            x5y5 = self.exp(landmark[:, cy, cx] * self.stride) #xxxxxyyyyy
            x5y5 = ((x5y5.reshape(2,5).transpose(1,0) + [cx,cy]) * self.stride) / [ratioW, ratioH]
            faceLandmarks[idx, :] = x5y5.flatten()

        keepIdx = self.nms(faceBoxes, 0.2)
        faceBoxes = faceBoxes[keepIdx, :]    
        faceLandmarks = faceLandmarks[keepIdx, :]
        
        return faceBoxes, faceLandmarks

    def faceAligner(self, inputImage, faceLandmarks=None, targetSize=(112,112)):
        alignFaces = np.empty((faceLandmarks.shape[0], targetSize[0], targetSize[1], 3), dtype=np.uint8)
        for bboxNo in range(faceLandmarks.shape[0]):
            faceLandmark = faceLandmarks[bboxNo].reshape(5,2)
            dst = faceLandmark.astype(np.float32)
            M = self.umeyama(dst, self.transDst)[0:2, :]
            alignFaces[bboxNo,:,:,:] = cv2.warpAffine(inputImage, M, (targetSize[1], targetSize[0]), borderValue=0.0)
        return alignFaces

    def detect(self, inputFrame, threshold=0.3):
        inputFrameRGB = cv2.cvtColor(inputFrame, cv2.COLOR_BGR2RGB)

        if self.processScale !=1:
            processFrame = cv2.resize(inputFrame,None, fx=self.processScale, fy=self.processScale)
        else:
            processFrame = inputFrame

        h, w = processFrame.shape[0], processFrame.shape[1]
        (hNew, wNew), (ratioH, ratioW) = self.calcImageScale(h, w)

        if len(processFrame.shape) != 3:
            processFrame = cv2.cvtColor(processFrame, cv2.COLOR_GRAY2BGR)
            
        processBlob = cv2.resize(processFrame, (wNew, hNew))
        processBlob = processBlob.transpose(2,0,1)[np.newaxis].astype(np.float32)
        
        cx, cy, scores, box, landmark = self.detector.run([], {'input': processBlob})
        faceBoxes, faceLandmarks = self.map2Box(cx, cy, scores, box, landmark, ratioW * self.processScale, ratioH * self.processScale, threshold=threshold)

        if len(faceBoxes) >= 1:
            alignedFace = self.faceAligner(inputFrame, faceLandmarks, targetSize=(112,112)) # RGB
            return faceBoxes, faceLandmarks, alignedFace
        else:
            return np.empty((0,5)), np.empty((0,10)), np.empty((0,112,112,3))

    def umeyama(self, src, dst, estimate_scale=True):
        num = src.shape[0]
        dim = src.shape[1]

        # Compute mean of src and dst.
        src_mean = src.mean(axis=0)
        dst_mean = dst.mean(axis=0)

        # Subtract mean from src and dst.
        src_demean = src - src_mean
        dst_demean = dst - dst_mean

        # Eq. (38).
        A = np.dot(dst_demean.T, src_demean) / num

        # Eq. (39).
        d = np.ones((dim,), dtype=np.double)
        if np.linalg.det(A) < 0:
            d[dim - 1] = -1

        T = np.eye(dim + 1, dtype=np.double)

        U, S, V = np.linalg.svd(A)

        # Eq. (40) and (43).
        rank = np.linalg.matrix_rank(A)
        if rank == 0:
            return np.nan * T
        elif rank == dim - 1:
            if np.linalg.det(U) * np.linalg.det(V) > 0:
                T[:dim, :dim] = np.dot(U, V)
            else:
                s = d[dim - 1]
                d[dim - 1] = -1
                T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))
                d[dim - 1] = s
        else:
            T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V.T))

        if estimate_scale:
            # Eq. (41) and (42).
            scale = 1.0 / src_demean.var(axis=0).sum() * np.dot(S, d)
        else:
            scale = 1.0

        T[:dim, dim] = dst_mean - scale * np.dot(T[:dim, :dim], src_mean.T)
        T[:dim, :dim] *= scale

        return T

