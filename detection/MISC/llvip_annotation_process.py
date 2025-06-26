import os
import json
import argparse
from tqdm import tqdm

def convert_coco_multimodal(input_json, base_root, output_json):
    """
    COCO 형식 val.json을 읽어,
      - file_name 필드를 제거하고
      - base_root 하위 visible/test, infrared/test 디렉터리 기준의 상대경로를
        img_path, img_ir_path에 기록
    다른 모든 필드는 그대로 유지합니다.
    """
    # load
    with open(input_json, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    # 절대 경로 루트 계산
    vis_root = os.path.join(base_root, 'visible', 'train')
    ir_root  = os.path.join(base_root, 'infrared', 'train')

    for img in tqdm(coco.get('images', []), desc='Updating images'):
        # 원래 file_name 가져오기
        fname = img.pop('file_name', None)
        if fname is None:
            raise KeyError(f"Missing 'file_name' in image entry with id={img.get('id')}")

        # 절대 경로를 상대경로로 변환
        abs_vis = os.path.join(vis_root, fname)
        abs_ir  = os.path.join(ir_root, fname)
        rel_vis = os.path.relpath(abs_vis, base_root)
        rel_ir  = os.path.relpath(abs_ir,  base_root)

        img['img_path']    = rel_vis.replace(os.path.sep, '/')    # 예: "visible/test/190001.jpg"
        img['img_ir_path'] = rel_ir.replace(os.path.sep, '/')     # 예: "infrared/test/190001.jpg"

    # save
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)

    print(f"Converted JSON saved to {output_json}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Convert COCO val.json to multimodal format with relative img_path/img_ir_path"
    )
    parser.add_argument('--input',     '-i', required=True,
                        help="원본 COCO JSON 파일 (e.g. /mnt/data/val.json)")
    parser.add_argument('--base_root', '-b', required=True,
                        help="LLVIP_coco 디렉터리 루트 (예: /media/jemo/HDD1/.../LLVIP_coco)")
    parser.add_argument('--output',    '-o', required=True,
                        help="결과 JSON 저장 경로 (e.g. /mnt/data/val_relpaths.json)")
    args = parser.parse_args()

    convert_coco_multimodal(
        input_json  = args.input,
        base_root   = args.base_root,
        output_json = args.output
    )
