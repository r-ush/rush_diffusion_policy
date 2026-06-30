import torch
import dill
import numpy as np
import cv2
import matplotlib.pyplot as plt
from diffusion_policy.workspace.base_workspace import BaseWorkspace

def visualize_resnet_heatmap(ckpt_path, image_path, output_path='heatmap.png'):
    # 1. 체크포인트 및 모델 로드
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    
    import hydra
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace.load_payload(payload)
    
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    policy.eval().cuda()

    # 2. 이미지 인코더 및 Spatial Softmax 레이어 찾아내기
    # Robomimic 구조상 보통 nets[0]은 backbone(ResNet), nets[1]은 pool(SpatialSoftmax)입니다.
    rgb_encoder = policy.obs_encoder.obs_nets['image0'] 
    resnet_backbone = rgb_encoder.nets[0]
    spatial_softmax = rgb_encoder.nets[1]

    # 3. 이미지 전처리
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    input_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().divide(255).unsqueeze(0).cuda()

    # 4. 특징 맵 추출 및 히트맵 계산
    with torch.no_grad():
        # ResNet 통과 (특징 맵 생성: 예 [1, 512, 7, 7])
        feature_map = resnet_backbone(input_tensor)
        
        # Spatial Softmax의 소프트맥스 맵(확률 분포) 가져오기
        # 이 부분이 바로 모델이 집중하고 있는 '히트맵'입니다.
        # SpatialSoftmax 레이어는 내부적으로 각 채널별로 가중치 맵을 계산합니다.
        heatmap = spatial_softmax._forward_libs(feature_map)['softmax'] # [1, N_keypoints, H, W]

    # 5. 히트맵 합치기 및 시각화
    # 모든 채널(Keypoint)의 히트맵을 하나로 합침 (평균 또는 최대값)
    combined_heatmap = torch.mean(heatmap[0], dim=0).cpu().numpy()
    
    # 원본 이미지 크기로 확대
    combined_heatmap = cv2.resize(combined_heatmap, (img.shape[1], img.shape[0]))
    combined_heatmap = (combined_heatmap - combined_heatmap.min()) / (combined_heatmap.max() - combined_heatmap.min())
    
    # 컬러맵 적용 (JET 효과)
    heatmap_color = cv2.applyColorMap(np.uint8(255 * combined_heatmap), cv2.COLORMAP_JET)
    
    # 원본 이미지와 합성
    overlay = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

    # 결과 저장 및 출력
    cv2.imwrite(output_path, overlay)
    print(f"히트맵이 {output_path}에 저장되었습니다.")
    
    # 화면 출력 (선택 사항)
    # plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    # plt.show()

if __name__ == "__main__":
    CKPT = "data/outputs/260429_0102_w_imp/epoch=0700-train_loss=0.002.ckpt"
    IMG = "test_image.jpg" # 실제 이미지 파일 경로로 수정하세요
    visualize_resnet_heatmap(CKPT, IMG)
