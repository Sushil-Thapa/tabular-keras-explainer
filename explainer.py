import os

from settings import SEED_VALUE, lime_out_path
os.environ['PYTHONHASHSEED']=str(SEED_VALUE)

import time
import numpy as np
import lime
import lime.lime_tabular
import random

def explain(model, data, args):
    print("Starting Explainer module...")
    n_instances = 5
    num_features = 3  # maximum number of features present in the explainations
    top_labels = 1  # number of max probable classes to consider

    (X_train, y_train, X_val, y_val, datatype_val, input_shape) = data

    print('Predicted label: ', np.argmax(model.predict(X_val), axis=1))
    
    start = time.time()
    print("Preparing tabular explainer...")
    
    feature_names = ['feature '+str(i) for i in range(args.n_feats)]
    class_names = ['class '+str(i) for i in range(y_train.shape[1])]

    explainer = lime.lime_tabular.LimeTabularExplainer(X_train, 
                                                    feature_names=feature_names, 
                                                    class_names=class_names, random_state=SEED_VALUE)
    print("Explainer preparation complete. Elapsed time:", time.time() - start)

    for i in range(n_instances):
        start = time.time()
        print("Starting explaining the instance...", i)
        
        print(f"\n{i}th sample")
        exp = explainer.explain_instance(X_val[i], model.predict_proba, num_features=num_features, \
                    top_labels=top_labels, num_samples=10)
        
        exp_avilable_labels = exp.available_labels()
        print("Number of labels to analyze",len(exp_avilable_labels))

        html_out = f"{lime_out_path}/{i}.html"
        print("Saving explainations to file",html_out)

        for l in exp_avilable_labels:
            print(f"\nsample {i}: class {l}: {class_names[l]}")
        #     display(pd.DataFrame(exp.as_list(label=i)))
            print(exp.as_list(label=l))

        exp.save_to_file(html_out)

    print("Explanation Iteration complete, Elapsed time:", time.time() - start)