import os
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    root_dir = "/home/CenteredData/TED Federated Learning Project/Photos"
    output_dir = "/data/michael/ted_rashmi/TED/data/TED"

    class1_pattern = "TED_1"  # Label 1
    class2_pattern = "CONT_"  # Label 0

    data = []
    for img_name in os.listdir(root_dir):
        if img_name.endswith(".png"):
            if class1_pattern in img_name:
                data.append({"directory": os.path.join(root_dir, img_name), "label": 1})
            elif class2_pattern in img_name:
                data.append({"directory": os.path.join(root_dir, img_name), "label": 0})

    # Convert to a DataFrame
    data_df = pd.DataFrame(data)

    # OPTION 1: Stratified Split (Maintains original ratio)
    train, temp = train_test_split(data_df, test_size=0.3, random_state=42, stratify=data_df["label"])
    val, test = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp["label"])

    # OPTION 2: Balanced Split (Equal TED_ and CONT_)
    # Limit TED_ to 62 samples to match CONT_
    # balanced_df = pd.concat([
    #     data_df[data_df['label'] == 1].sample(n=62, random_state=42),
    #     data_df[data_df['label'] == 0]
    # ], ignore_index=True)
    #
    # train, temp = train_test_split(balanced_df, test_size=0.3, random_state=42, stratify=balanced_df["label"])
    # val, test = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp["label"])

    os.makedirs(output_dir, exist_ok=True)
    train.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    print("Train, validation, and test CSV files generated!")

if __name__ == "__main__":
    main()
