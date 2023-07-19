import numpy as np

def calcAccuracy(LPred, LTrue):
    """Calculates prediction accuracy from data labels.

    Args:
        LPred (array): Predicted data labels.
        LTrue (array): Ground truth data labels.

    Retruns:
        acc (float): Prediction accuracy.
    """

    # --------------------------------------------
    # === Your code here =========================
    # --------------------------------------------
    tcorrect = 0
    for i in range(len(LTrue)):
        if LTrue[i] == LPred[i]:
            tcorrect += 1
    acc = tcorrect / float(len(LTrue)) * 100.0
    # ============================================
    return acc


def calcConfusionMatrix(LPred, LTrue):
    """Calculates a confusion matrix from data labels.

    Args:
        LPred (array): Predicted data labels.
        LTrue (array): Ground truth data labels.

    Returns:
        cM (array): Confusion matrix, with predicted labels in the rows
            and actual labels in the columns.
    """

    # --------------------------------------------
    # === Your code here =========================
    # --------------------------------------------
    classes = np.unique(LTrue)
    NClasses = classes.shape[0]
    cM = np.zeros((NClasses, NClasses))
    
    for LTrue, LPred in zip(LTrue, LPred):
        if LTrue == LPred:
            cM[LTrue-1][LTrue-1] += 1
        else:
            cM[int(LPred)-1][int(LTrue)-1] += 1
        
    # ============================================

    return cM


def calcAccuracyCM(cM):
    """Calculates prediction accuracy from a confusion matrix.

    Args:
        cM (array): Confusion matrix, with predicted labels in the rows
            and actual labels in the columns.

    Returns:
        acc (float): Prediction accuracy.
    """

    # --------------------------------------------
    # === Your code here =========================
    # --------------------------------------------
    acc = sum(np.diag(cM)) / sum(sum(cM))
    # ============================================
    
    return acc
