# %% [markdown]
# ---
# # Setting environment

# %%
# !pip install scipy==1.13.1
# !pip install hmmlearn
# !pip install numpy==1.26.4
# !pip install pandas==2.1.4
# !pip install matplotlib==3.7.1
# !pip install scikit-learn==1.3.2
# !pip install networkx==3.3
# !pip install -u ucimlrepo
# !pip install openpyxl

import numpy as np
import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial.distance import euclidean
import networkx as nx
from hmmlearn import hmm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform

import random
import math
import pickle

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

def standardise_dataset(D):

    D_std = stats.zscore(D, axis=0, ddof = 1) #standardise the dataset and degree of freedom = n - 1
    
    return D_std, D.mean(), D.std()

def standardise_multiple_dataset(D):

    D_std = {}
    D_mean = {}
    D_sd = {}
    
    for key,value in D.items():
        D_std[key], D_mean[key], D_sd[key]  = standardise_dataset(value)
        
    return D_std, D_mean, D_sd

def create_pseudo_time_series(D, C, T, k, i):
    
    np.random.seed(i)
    D_array = D.to_numpy()
    C_array = C["Diag"].to_numpy()

    P = []  
    original_indices = []  

    for _ in range(k):
        while True:
            d_i_indices = np.random.choice(len(D), T, replace=True)
            if len(np.unique(C_array[d_i_indices])) > 1:  # Ensure at least one healthy and one diseased class
                break

        d_i = D_array[d_i_indices]
        c_i = C_array[d_i_indices]

        # Select start and end indices
        healthy_indices = np.where(c_i == 0)[0]
        diseased_indices = np.where(c_i == 1)[0]
        start = np.random.choice(healthy_indices)
        end = np.random.choice(diseased_indices)

        # Construct distance matrix W_i
        W_i = squareform(pdist(d_i))

        # Order d_i using Floyd-Warshall algorithm
        G = nx.from_numpy_array(W_i)
        path = nx.shortest_path(G, source=start, target=end, weight="weight")

        # Ensure all indices are included in the path
        missing_indices = set(range(T)) - set(path)
        for idx in missing_indices:
            insert_costs = [W_i[path[i], idx] + W_i[idx, path[i+1]] for i in range(len(path)-1)]
            insert_pos = np.argmin(insert_costs)
            path.insert(insert_pos + 1, idx)

        # Add ordered d_i* to P
        d_i_star = d_i[path]
        pseudo = pd.DataFrame(d_i_star, columns=D.columns)
        pseudo["Diag"] = c_i[path]
        P.append(pseudo)
        
        # Store the original indices for this pseudo time-series
        original_indices.append(d_i_indices[path])

    return P, original_indices

def multi_pseudo_time_series(lengths, no_ts_list, X, y, seed):
    
    multi_series = {}
    order = {}
    
    for T in lengths:
        for k in no_ts_list:
            key = f"{T:03d},{k:03d}"  # This ensures 3-digit format for both T and k
            multi_series[key], order[key] = create_pseudo_time_series(X, y, T, k, seed)
    
    return multi_series, order

def split_dataframes(P, no_of_row: int): #splitting the dataset into two parts

    if isinstance(P, dict):
        dict1 = {}
        dict2 = {}

        for key, values in P.items():
            if isinstance(values, list):
                list1 = []
                list2 = []

                for df in values:
                    ls1 = df.iloc[:, :-no_of_row]
                    ls2 = df.iloc[:, -no_of_row:]
                    list1.append(ls1)
                    list2.append(ls2)

                dict1[key] = list1
                dict2[key] = list2

            elif isinstance(values, pd.DataFrame):
                dict1[key] = values.iloc[:, :-no_of_row]
                dict2[key] = values.iloc[:, -no_of_row:]

        return dict1, dict2
    
    elif isinstance(P, list):
        list1 = []
        list2 = []
        for df in P:
            if isinstance(df, pd.DataFrame):
                ls1 = df.iloc[:, :-no_of_row]
                ls2 = df.iloc[:, -no_of_row:]
                list1.append(ls1)
                list2.append(ls2)
                
        return list1, list2

def single_dataframe(df_list, reset_index=True, add_source_column=False, dropping = True):
    
    if add_source_column:
        for i, df in enumerate(df_list):
            df = df.copy()
            df["source"] = f"df_{i}"
            df_list[i] = df
    
    merged_df = pd.concat(df_list, axis=0, ignore_index=reset_index)
    
    if dropping:
        merged_df = merged_df.drop_duplicates(keep = "last")
    if reset_index == True:
        merged_df = merged_df.reset_index(drop = True)
        
    return merged_df

def single_dataframe_dict(P, dropping = True):
    new_dict = {}
    for key,value in P.items():
        new_dict[key] = single_dataframe(value, dropping=dropping)

    return new_dict

def extract_specific_columns(data_dict, column_names):

    result = {}
    
    # Ensure column_names is a list
    if isinstance(column_names, str):
        column_names = [column_names]
    
    for key, value in data_dict.items():
        # Extract the specified columns
        result[key] = value[column_names].copy()
    
    return result


def train_hmm_dict(data: str, P: dict, n_states: int, seed: int,  n_iterations: int, hidden_columns_name = "hidden"):
    np.random.seed(seed)
    hidden = {}
    models = {}

    for key, value in P.items():
        print(f"Now start training data {data}: {key}")
        X = np.vstack([df for df in value])
    
        # Get the lengths of each time series
        lengths = [len(df) for df in value]
    
        model = hmm.GaussianHMM(n_components= n_states, covariance_type="full", n_iter= n_iterations, random_state=seed)

        model.fit(X, lengths=lengths)

        print("Training End!")

        hidden_seq = model.predict(X, lengths=lengths)
        split_hidden = np.array_split(hidden_seq, len(value))
        hidden_df = [pd.DataFrame(array, columns=[hidden_columns_name]) for array in split_hidden]

        models[key] = model
        hidden[key] = hidden_df

    return models, hidden

# # Check whther any hidden state match with the diagnosis

def merging_dict(*dicts):

    keys = dicts[0].keys()
    merged_dict = {}
    for key in keys:
        dfs_list = [d[key] for d in dicts]
        merged_dict[key] = [pd.concat(dfs, axis=1) for dfs in zip(*dfs_list)]
    
    return merged_dict

def check_state(P: dict,col1: str,col2: str, order = None):
    hs_matchings = {}
    
    for key, value in P.items():
        print(f" counting {key}:")
        aggregated_df = pd.concat(value)
        aggregated_df = aggregated_df.drop_duplicates(keep="last")
        print(f"The data at {key} has shape {aggregated_df.shape}")
        
        hs_match = aggregated_df.groupby(col1)[col2].value_counts().unstack(fill_value=0)
        hs_match["Total"] = hs_match.sum(axis=1)
        if order is not None:
            hs_match.index = order[key]
            hs_match.sort_index(inplace=True)
        
        hs_matchings[key] = hs_match
    
    return hs_matchings

def save_dict_dataframes_to_excel(dict_of_dfs, excel_path):

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for key, value in dict_of_dfs.items():
            value.to_excel(writer, sheet_name=key, index=False)
            
    print(f"DataFrames have been saved to {excel_path}")


# ---
# # Visualise the code

def plotting_trajectory(data, target_col,title,  
                        n_components=2,result_name = "Diag",
                        rename_order=None, colors=None,
                        trajectory_indices=None,
                        line_color="gray", start_color="green", end_color="purple",
                        start_end_size=50, save_path=None,
                        figsize=(8, 6)
                        ):
    
    if isinstance(data, pd.DataFrame):
        df = data.copy()
        obs = df.drop(columns=[target_col])
        result = df[target_col]

    elif isinstance(data, tuple) and len(data) == 2:
        obs, result = data
        df = pd.concat([obs, result], axis=1)

    unique_values = sorted(df[target_col].unique())
    
    if rename_order is not None:

        rename_order = [rename -1 for rename in rename_order]
        rename_dict = {unique_values[i]: f"State {rename_order[i] + 1}" for i in range(len(rename_order))}
        print("Rename dictionary:", rename_dict)
        df[target_col] = df[target_col].map(rename_dict)
        
        unique_values = [f"State {i + 1}" for i in range(len(rename_order))]
    

    if colors is not None:
        if len(colors) > len(unique_values):
            colors = colors[:len(unique_values)]
        color_dict = dict(zip(unique_values, colors))
    else:
        color_dict = None
    
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(obs)
    
    pca_df = pd.DataFrame(data=X_pca, columns=[f"PC{i+1}" for i in range(n_components)])
    pca_df[target_col] = df[target_col]
    pca_df[result_name] = result
    
    if rename_order is not None:
        pca_df[target_col] = pd.Categorical(pca_df[target_col], 
                                            categories=unique_values,
                                            ordered=True)
        pca_df = pca_df.sort_values(target_col)
    
    plt.figure(figsize=figsize)
    
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue=target_col, palette=color_dict, alpha=0.6)
    
    if trajectory_indices is not None:

        trajectory_points = X_pca[trajectory_indices]
        
        plt.plot(trajectory_points[:, 0], trajectory_points[:, 1], c=line_color, alpha=0.7, linestyle="--", label="Trajectory")
        
        for i in range(len(trajectory_points) - 1):
            plt.annotate("", xy=(trajectory_points[i+1, 0], trajectory_points[i+1, 1]), 
                         xytext=(trajectory_points[i, 0], trajectory_points[i, 1]),
                         arrowprops=dict(arrowstyle="->", color=line_color, alpha=0.7))
        
        plt.scatter(trajectory_points[0, 0], trajectory_points[0, 1], c=start_color, s=start_end_size, 
                    label="Start", edgecolor="black")
        plt.scatter(trajectory_points[-1, 0], trajectory_points[-1, 1], c=end_color, s=start_end_size, 
                    label="End", edgecolor="black")
    
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    
    plt.show()


def reorder(A: list):

    B = [i + 1 for i in range(len(A))]
    C = sorted(list(zip(A, B)))
    D = [C[i][1] for i in range(len(C))]

    return D

def rearrange_matrix(matrix, orders):

    new_matrix = np.zeros_like(matrix)
    new_orders = reorder(orders)
    new_orders = [order - 1 for order in new_orders]

    for i, new_i in enumerate(new_orders):
        for j, new_j in enumerate(new_orders):
            new_matrix[new_i, new_j] = matrix[i, j]
    
    return new_matrix

def transitional_matrix(model, filename,order_dict = None):

    transitional_matrix = {}
    for key, value in model.items():
        transitional_matrix[key] = np.round(value.transmat_,3)

    order_matrix = {}
    for key,value in transitional_matrix.items():
        if order_dict is None:
            order_matrix[key]= value
        else:
            order_matrix[key]= rearrange_matrix(value,order_dict[key])

    with open(filename, "w") as file:
        for key, value in order_matrix.items():
            file.write(f"{key}:\n")
            for row in value:
                file.write("  " + " ".join(map(str, row)) + "\n")
            file.write("\n")  #
            
    return order_matrix

def plot_mean_by_hidden(df, title, grouping="hidden", ignore_col=None, figsize=(10, 6), rename_order=None, save_fig=True):

    if isinstance(ignore_col, str):
        ignore_col = [ignore_col]
    elif ignore_col is None:
        ignore_col = []
        
    columns_to_plot = [col for col in df.columns if col != grouping and col not in ignore_col]
    
    grouped_means = df.groupby(grouping)[columns_to_plot].mean()
    
    if rename_order is not None:
        grouped_means.index = rename_order
        grouped_means.sort_index(inplace=True)

    fig, ax = plt.subplots(figsize=figsize)
    
    for hidden_value, data in grouped_means.iterrows():
        valid_data = data.dropna()
        valid_columns = valid_data.index.tolist()
        
        if len(valid_data) > 0:
            if rename_order is None:
                label = f"{hidden_value + 1} {grouping}"
            else:
                label = f"{hidden_value} {grouping}"
            ax.plot(valid_columns, valid_data.values, marker="o", label=label)
    
    ax.set_title(f"{title} Grouped by {grouping}")
    ax.set_xlabel("Features", fontsize=15)
    ax.set_ylabel("Mean Value", fontsize=15)
    ax.legend()
    ax.grid(True)

    plt.xticks(rotation=45, ha="right")
    
    plt.tight_layout()

    if save_fig:
        file_name = title.replace(" ", "_") + ".png"
        plt.savefig(file_name, dpi=300, bbox_inches="tight")
        print(f"Figure saved as {file_name}")

    return fig

def plot_mean_by_hidden_dict(P, title,ignore_col = None, grouping="hidden", rename_order = None):
    
    for key,value in P.items():
        plot_mean_by_hidden(value,f"{title} {key}", ignore_col=ignore_col, grouping=grouping, rename_order = rename_order[key] if rename_order is not None else None) 

    return None


# Generating New Pseudo-time seris

def generate_time_series(model, length, n_series, feature_names, random_seed=42):

    np.random.seed(random_seed)
    
    dfs = []
    for i in range(n_series):
      
        X, hid = model.sample(length, random_state=i)
        
        noise = np.random.normal(0, 0.1, X.shape)
        X_noisy = X + noise
        
        df = pd.DataFrame(X_noisy, columns=feature_names)
        df["series_id"] = i
        df["time_step"] = range(length)
        df["hidden"] = hid
        dfs.append(df)

    return dfs


def gen_multi_ts(P,title):

    gen_TS = {}
    for key, value in P.items():
        gen_TS[key] = generate_time_series(value, int(key[:3]), int(key[4:]),title, random_seed=42)
        
    return gen_TS


def destandardise(P, mean, std):

    destandardised = {}
    for key, value in P.items():
        series = []
        for i in range(len(value)):
            D = value[i]*std + mean
            series.append(D)
        destandardised[key] = series
        
    return destandardised



# # Train a classifier to see the generated data are classified as Dieseases or Healthy


def compare_classifiers(X, y, data_name, standardize=True):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if standardize:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test
        scaler = None
        
    classifiers = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(),
        "Naive Bayes": GaussianNB()
    }
    
    best_accuracy = 0
    best_classifier = None
    best_classifier_name = ""
    
    for key, value in classifiers.items():
        value.fit(X_train_scaled, y_train)
        y_pred = value.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nFor {data_name} - {key}:")
        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_classifier = value
            best_classifier_name = key
    
    print(f"For {data_name}, the most accurate classifier is {best_classifier_name} with an accuracy of {best_accuracy:.4f}")
    
    return best_classifier, scaler, accuracy, best_classifier_name


# # Classifier the generated data

def disease_prediction(P,classifier,scaler,col_name):

    predictions = {}
    for key,value in P.items():
        prediction = []
        for i in range(len(value)):
            prediction.append(pd.DataFrame({
                col_name:classifier.predict(scaler.fit_transform(value[i]))
                }))
        predictions[key]=prediction
        
    return predictions


# Obtaining the cross-sectional data


def cross_sectional(combined_dict, num_rows, required_hidden_states=None):
    np.random.seed(42)
    output_dict = {}
    for key, value in combined_dict.items():
        num_dfs = int(key.split(",")[1])  
        
        sampled_rows = []
        rows_per_df = num_rows // num_dfs
        remaining_rows = num_rows % num_dfs
        
        for df_index, df in enumerate(value):
            rows_to_sample = rows_per_df + (1 if df_index < remaining_rows else 0)
            
            sample_indices = np.random.choice(len(df), size=min(rows_to_sample, len(df)), replace=False)
            
            sample = df.iloc[sample_indices].copy()
            
            sample["source_df"] = df_index
            
            sampled_rows.extend(sample.to_dict("records"))
        
        new_df = pd.DataFrame(sampled_rows)
        
        if len(new_df) > num_rows:
            new_df = new_df.sample(num_rows)
        elif len(new_df) < num_rows:
            additional_rows = num_rows - len(new_df)
            extra_sample = new_df.sample(additional_rows, replace=True)
            new_df = pd.concat([new_df, extra_sample], ignore_index=True)
        
        has_diag_0_and_1 = set([0, 1]).issubset(new_df["Diag"].unique())
        

        if required_hidden_states is None:
            required_hidden_states = set(new_df["hidden"].unique())
        
        has_required_hidden = required_hidden_states.issubset(new_df["hidden"].unique())
        
        if has_diag_0_and_1 and has_required_hidden:
            output_dict[key] = new_df
        
    return output_dict


# See the generated data structure


def create_confusion_matrix(df, col1, col2, rename_order = None):

    conf_matrix = pd.crosstab(df[col1], df[col2])
    
    all_values = sorted(set(df[col1].unique()) | set(df[col2].unique()))
    for val in all_values:
        if val not in conf_matrix.index:
            conf_matrix.loc[val] = 0
        if val not in conf_matrix.columns:
            conf_matrix[val] = 0
    
    conf_matrix = conf_matrix.sort_index().sort_index(axis=1)
    conf_matrix.columns = [i + 1 for i in range(len(df[col1].unique()))]
    if rename_order is None:
        conf_matrix.index = [i + 1 for i in range(len(df[col1].unique()))]
    else:
        conf_matrix.index = rename_order
        conf_matrix = conf_matrix.sort_index().sort_index(axis=1)
        
    conf_matrix = conf_matrix.fillna(0)
    
    return conf_matrix

def plot_transitional_matrix(transitional_matrix, title):
    labels = [i + 1 for i in range(len(transitional_matrix))]
    
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(transitional_matrix, annot=True, cmap="Blues", ax=ax, xticklabels=labels, yticklabels=labels, vmin = 0, vmax = 1)
    ax.set_title(title)
    ax.set_xlabel("To")
    ax.set_ylabel("From")
    
    plt.tight_layout()
    plt.show()


def plot_transitional_matrix_dict(P, title):

    for key, value in P.items():
        labels = [i + 1 for i in range(len(value))]
    
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(value, annot=True, cmap="Blues", ax=ax, xticklabels=labels, yticklabels=labels,  vmin = 0, vmax = 1)
        ax.set_title(f"Transitional Matrix for {title} {key}")
        ax.set_xlabel("To")
        ax.set_ylabel("From")
        
        plt.tight_layout()
        plt.show()

def compare_single_matrices(matrix_dict, comparison_matrix, matrix_name=None, comparison_matrix_name="comparison matrix"):
    if isinstance(matrix_dict, np.ndarray):
        matrix_dict = {matrix_name: matrix_dict}
    labels = [i + 1 for i in range(len(comparison_matrix))]
    
    for key, value in matrix_dict.items():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        sns.heatmap(value, annot=True, cmap="Blues", ax=ax1, xticklabels=labels, yticklabels=labels, vmin=0, vmax=1)
        ax1.set_title(key)
        ax1.set_xlabel("To")
        ax1.set_ylabel("From")
        
        sns.heatmap(comparison_matrix, annot=True, cmap="Blues", ax=ax2, xticklabels=labels, yticklabels=labels, vmin=0, vmax=1)
        ax2.set_title(comparison_matrix_name)
        ax2.set_xlabel("To")
        ax2.set_ylabel("From")
        
        plt.tight_layout()
        plt.show()

def compare_matrices_dict(matrix_dict, comparison_dict, title = "New", compare_title = "Comparison"):
    
    for key, matrix1 in matrix_dict.items():
        if key not in comparison_dict:
            continue
        
        matrix2 = comparison_dict[key]
        labels = [i + 1 for i in range(len(matrix1))]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        sns.heatmap(matrix1, annot=True, cmap="Blues", ax=ax1, xticklabels=labels, yticklabels=labels, vmin = 0, vmax = 1)
        ax1.set_title(f"{title}: {key}")
        ax1.set_xlabel("To")
        ax1.set_ylabel("From")
        
        sns.heatmap(matrix2, annot=True, cmap="Blues", ax=ax2, xticklabels=labels, yticklabels=labels, vmin = 0, vmax = 1)
        ax2.set_title(f"{compare_title}: {key} :")
        ax2.set_xlabel("To")
        ax2.set_ylabel("From")
        
        plt.tight_layout()
        plt.show()

def abs_diff_matrix_dict(matrix_list1,matrix):

    abs_error = {}
    for key,value in matrix_list1.items():
        abs_error[key]= np.round(abs(value - matrix).sum(),3)
        print(f"{key}: {np.round(abs(value - matrix).sum(),3)}")
        
    return abs_error

def plot_scatter_for_bias(data_dict):

    keys = list(data_dict.keys())
    values = list(data_dict.values())

    plt.figure(figsize=(10 , 10))
    plt.scatter(range(len(keys)), values, color="blue")

    plt.xlabel("Different datasets")
    plt.ylabel("Absolutes difference with the original tansitional matrices")
    plt.title("Scatter Plot of the differnce of transitional matrices")
    plt.xticks(range(len(keys)), keys, rotation=45, ha="right")

    for i, value in enumerate(values):
        plt.annotate(f"{value:.3f}", (i, value), textcoords="offset points", xytext=(0,10), ha="center")

    plt.tight_layout()
    plt.show()

