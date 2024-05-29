def main():
    model = YOLO("yolov8n.pt", task="detect")
    metrics = model.val()
    print(metrics)


def prin():
    n = {
        "metrics/precision(B)": 0.5414162308013722,
        "metrics/recall(B)": 0.47706422018348627,
        "metrics/mAP50(B)": 0.4651260164006293,
        "metrics/mAP50-95(B)": 0.15814004479113142,
        "fitness": 0.18883864195208122,
    }
    s = {
        "metrics/precision(B)": 0.6264528662424486,
        "metrics/recall(B)": 0.518348623853211,
        "metrics/mAP50(B)": 0.5385276597059194,
        "metrics/mAP50-95(B)": 0.19026044875301723,
        "fitness": 0.22508716984830746,
    }
    l = {
        "metrics/precision(B)": 0.44397393479543373,
        "metrics/recall(B)": 0.43119266055045874,
        "metrics/mAP50(B)": 0.3121359448785275,
        "metrics/mAP50-95(B)": 0.09143586934295116,
        "fitness": 0.1135058768965088,
    }
    m = {
        "metrics/precision(B)": 0.3827781417251571,
        "metrics/recall(B)": 0.5275229357798165,
        "metrics/mAP50(B)": 0.3451034012527603,
        "metrics/mAP50-95(B)": 0.10237087965372293,
        "fitness": 0.1266441318136267,
    }
    x = {
        "metrics/precision(B)": 0.3827781417251571,
        "metrics/recall(B)": 0.5275229357798165,
        "metrics/mAP50(B)": 0.3451034012527603,
        "metrics/mAP50-95(B)": 0.10237087965372293,
        "fitness": 0.1266441318136267,
    }

    ren = {
        "metrics/precision(B)": 0.6902122553532517,
        "metrics/recall(B)": 0.7339449541284404,
        "metrics/mAP50(B)": 0.6978231866221873,
        "metrics/mAP50-95(B)": 0.2768027388514156,
        "fitness": 0.3189047836284928,
    }
    res = {
        "metrics/precision(B)": 0.5874216054582164,
        "metrics/recall(B)": 0.7247706422018348,
        "metrics/mAP50(B)": 0.6673627876202782,
        "metrics/mAP50-95(B)": 0.2716520410746606,
        "fitness": 0.3112231157292224,
    }
    rel = {
        "metrics/precision(B)": 0.6320874479880857,
        "metrics/recall(B)": 0.555045871559633,
        "metrics/mAP50(B)": 0.5902226358236523,
        "metrics/mAP50-95(B)": 0.24912735332724653,
        "fitness": 0.2832368815768871,
    }
    rem = {
        "metrics/precision(B)": 0.6573764482274,
        "metrics/recall(B)": 0.6651376146788991,
        "metrics/mAP50(B)": 0.6482527443794568,
        "metrics/mAP50-95(B)": 0.24832120980556388,
        "fitness": 0.2883143632629532,
    }
    rex = {
        "metrics/precision(B)": 0.6645329065873431,
        "metrics/recall(B)": 0.6972477064220184,
        "metrics/mAP50(B)": 0.6639770173033037,
        "metrics/mAP50-95(B)": 0.25329698361319336,
        "fitness": 0.2943649869822044,
    }

    dik = rex

    for key in dik:
        print(dik[key], end=" ")


prin()
