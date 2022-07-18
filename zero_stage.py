import glob
import joblib

img_list = glob.glob('./test/*')
preds, new_preds, statuses = [], [], []
for fl in img_list:
    preds.append((fl, 10496 // 2, 10496 // 2, 180.0))
    new_preds.append((fl, 10496 // 2, 10496 // 2, 180.0))
    statuses.append(False)

d = {
    'preds': preds,
    'new_preds': new_preds,
    'statuses': statuses
}
joblib.dump(d, 'preds.joblib')