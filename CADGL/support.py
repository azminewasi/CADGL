import datetime
import os
import numpy as np
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.manifold import TSNE
from sklearn.cluster import SpectralClustering

def support_tasks(model,latent_info_encoder,df):
    now = datetime.datetime.now()
    currentDateAndTime=now.strftime("%Y-%m-%d-%H-%M-%S")
    currentDateAndTime="./Experimental_Scores/"+currentDateAndTime
    path="./"+currentDateAndTime+"/"
    os.mkdir(path)
    
    z = latent_info_encoder["drug"].cpu().numpy()
    np.save("./"+currentDateAndTime+"/"+"drug_embedding.npy", z)
    
    df.to_csv("./"+currentDateAndTime+"/"+"scores.csv")
    
    
    named_layers = dict(model.named_modules())
    dict_str = str(named_layers)
    
    with open("./"+currentDateAndTime+"/"+"model_dict.txt", "w") as f:
        f.write(dict_str)
        
        
    zx = np.load("./"+currentDateAndTime+"/"+"drug_embedding.npy")
    z_embed = TSNE(n_components=2, learning_rate='auto', init='random', 
                   perplexity=3).fit_transform(zx)
    plt.scatter(z_embed[:, 0], z_embed[:, 1])
    plt.axis("off")
    plt.savefig("./"+currentDateAndTime+"/"+"tnse.png", bbox_inches = "tight", transparent = True)