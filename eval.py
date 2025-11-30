import os
import cv2
import py_sod_metrics
import argparse

FM = py_sod_metrics.Fmeasure()
WFM = py_sod_metrics.WeightedFmeasure()
SM = py_sod_metrics.Smeasure()
EM = py_sod_metrics.Emeasure()
MAE = py_sod_metrics.MAE()
MSIOU = py_sod_metrics.MSIoU()

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, required=True, 
                    help="path to the prediction results")
parser.add_argument("--pred_path", type=str, required=True, 
                    help="path to the prediction results")
parser.add_argument("--gt_path", type=str, required=True,
                    help="path to the ground truth masks")
args = parser.parse_args()

sample_gray = dict(with_adaptive=True, with_dynamic=True)
FMv2 = py_sod_metrics.FmeasureV2(
    metric_handlers={
        "fm": py_sod_metrics.FmeasureHandler(**sample_gray, beta=0.3),
        "f1": py_sod_metrics.FmeasureHandler(**sample_gray, beta=1),
        "iou": py_sod_metrics.IOUHandler(**sample_gray),
        "dice": py_sod_metrics.DICEHandler(**sample_gray),
    }
)

pred_root = args.pred_path
mask_root = args.gt_path
mask_name_list = sorted(os.listdir(mask_root))
for i, mask_name in enumerate(mask_name_list):
    print(f"[{i}] Processing {mask_name}...")
    mask_path = os.path.join(mask_root, mask_name)
    pred_path = os.path.join(pred_root, mask_name[:-4] + '.png')
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

    FM.step(pred=pred, gt=mask)
    WFM.step(pred=pred, gt=mask)
    SM.step(pred=pred, gt=mask)
    EM.step(pred=pred, gt=mask)
    MAE.step(pred=pred, gt=mask)
    FMv2.step(pred=pred, gt=mask)
    

fm = FM.get_results()["fm"]
wfm = WFM.get_results()["wfm"]
sm = SM.get_results()["sm"]
em = EM.get_results()["em"]
mae = MAE.get_results()["mae"]
fmv2 = FMv2.get_results()

curr_results = {
    "meandice": fmv2["dice"]["dynamic"].mean(),
    "meaniou": fmv2["iou"]["dynamic"].mean(),
    'Smeasure': sm,
    "wFmeasure": wfm,  # For Marine Animal Segmentation, Dichotomous Image Segmentation
    "adpFm": fm["adp"], # For Camouflaged Object Detection
    "meanFm": fmv2['fm']['dynamic'].mean(), # For Remote Sensing Saliency Detection
    "maxFm": fmv2['fm']['dynamic'].max(), # For Remote Sensing Saliency Detection
    "meanEm": em["curve"].mean(),
    "MAE": mae,
}

print(args.dataset_name)
print("mDice:          ", format(curr_results['meandice'], '.3f'))
print("mIoU:           ", format(curr_results['meaniou'], '.3f'))
print("S_{alpha}:      ", format(curr_results['Smeasure'], '.3f'))
print("F^{w}_{beta}:   ", format(curr_results['wFmeasure'], '.3f'))
print("F_{beta}:       ", format(curr_results['adpFm'], '.3f'))
print("F^{mean}_{beta}:", format(curr_results['meanFm'], '.3f'))
print("F^{max}_{beta}: ", format(curr_results['maxFm'], '.3f'))
print("E_{phi}:        ", format(curr_results['meanEm'], '.3f'))
print("MAE:            ", format(curr_results['MAE'], '.3f'))