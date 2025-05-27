from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
import pandas as pd
import os

def save_cv_metrics(y_true, y_pred, classes, outdir="metrics"):
    """Save classification metrics to CSV files for cross-validation"""
    os.makedirs(outdir, exist_ok=True)
    
    # Get current fold index
    acc_file = f"{outdir}/accuracy.csv"
    index = 0
    if os.path.exists(acc_file):
        index = len(pd.read_csv(acc_file))
    
    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    
    # Print results for reference
    print(f"Fold {index} - Acc: {acc:.4f}")
    print(classification_report(y_true, y_pred, target_names=classes))
    print(f"Confusion Matrix:\n{cm}")
    
    # Save accuracy
    pd.DataFrame({'fold': [index], 'accuracy': [acc]}).to_csv(
        acc_file, 
        mode='a', 
        header=not os.path.exists(acc_file),
        index=False
    )
    
    # Save classification report
    metrics = []
    for cls in list(report.keys()):
        if cls in ['accuracy', 'samples avg']:
            continue
        row = report[cls] if isinstance(report[cls], dict) else {'value': report[cls]}
        row['class'] = cls
        row['fold'] = index
        metrics.append(row)
    
    report_file = f"{outdir}/classification_report.csv"
    pd.DataFrame(metrics).to_csv(
        report_file,
        mode='a',
        header=not os.path.exists(report_file),
        index=False
    )
    
    # Save confusion matrix
    pd.DataFrame(
        cm, 
        index=[f'True_{c}' for c in classes],
        columns=[f'Pred_{c}' for c in classes]
    ).assign(fold=index).to_csv(f"{outdir}/cm_fold_{index}.csv")
    
    return acc

def get_cv_averages(outdir="metrics"):
    """Calculate average metrics across all folds"""
    # Avg accuracy
    acc = pd.read_csv(f"{outdir}/accuracy.csv")['accuracy'].mean()
    
    # Avg class metrics 
    report = pd.read_csv(f"{outdir}/classification_report.csv")
    avg_report = report.groupby('class').mean().reset_index()
    avg_report.to_csv(f"{outdir}/avg_report.csv", index=False)
    
    print(f"Average accuracy: {acc:.4f}")
    return acc, avg_report


def evaluate_testset(trainer, model, datamodule, device='cuda', save_path=''):
    """
    Evaluate model on the test set. Prints accuracy, classification report, confusion matrix.
    """
    # Load test data
    test_loader = datamodule.test_dataloader()

    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    img_paths = []
    pred_probs = []

    with torch.no_grad():
        for batch in test_loader:
            imgs, labels = batch['img'], batch['label']
            imgs, labels = imgs.to(device), labels.to(device)

            logits = model(imgs)
            
            probs = torch.nn.functional.softmax(logits, dim=1)
            pred_prob, preds = torch.max(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            img_paths.extend(batch['img_path'])
            pred_probs.extend(pred_prob.cpu().numpy())
   

    # Create a DataFrame from the dictionary
    df = pd.DataFrame({'img_paths': img_paths, 'labels': all_labels, 'preds': all_preds, 'pred_probs': pred_probs})
    df.to_csv(f'{trainer.logger.log_dir }/test_results.csv', index=False)


    save_cv_metrics(all_labels, all_preds, list(datamodule.classes.keys()), outdir=save_path)

    '''
    # Compute accuracy
    acc = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {acc * 100:.2f}%")

    # Classification report
    classes_list = list(datamodule.classes.keys())  # e.g. ["TED_1", "CONT_"]
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes_list))

    # Confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    '''