import matplotlib.pyplot as plt
import os

def get_data():

    folder_epoch = {
        0:{'folder':'Results2', 'round':70}, 
        1:{'folder':'Results3', 'round': 110}, 
        2:{'folder':'Results4', 'round': 150}, 
        3:{'folder':'Results5', 'round':200}, 
    }
    figure_dict = {
        'Selection_attack_succ.png':'Figure 6 a',
        'adversary_count_attack_succ.png':'Figure 6 b Random part',
        'adversary_count_pagerank_attack_succ.png':'Figure 6 b PageRank part',
        'adversary_count.png':'Figure 6b',
        'GraphType_pagerank_attack_succ.png':'Figure 6 c',
        'node_count.png':'Figure 7 a',
        'Dataset_pagerank_attack_succ.png':' Figure 7 b',
        'Fault_tolerant_attack_succ.png':' Figure 7 c',
        'TrimmedMean_bar_attack_succ.png':'Figure 8',
        'Clipping_bar_attack_succ.png':'Figure 9 a paper already have 200 epochs shown on figure left-top',
        'Clipping_bar_test_acc.png':'Figure 9 b paper already have 200 epochs shown on figure left-top',
        'OurDefense_attack_succ.png':'Figure 9 c paper already have 200 epochs shown on figure left-top',
    }
    combined_folder = 'Results_Combined'
    return folder_epoch, figure_dict, combined_folder 

def combine_figures():
    folder_epoch, figure_dict, combine_folder = get_data()
    
    for figure_name, description in figure_dict.items():
            
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))

        for i, axi in enumerate(ax.flat):
            details = folder_epoch[i]
            folder = details['folder']
            epoch_count = details['round']
            img_full_path =  os.path.join(folder, figure_name)
            img = plt.imread(img_full_path)
            axi.set_title(f'Round: {epoch_count}')
            axi.imshow(img)
        
        fig.suptitle(f'{figure_name} Description: {description}')
        fig.savefig(os.path.join(combine_folder, figure_name))

def main():
    combine_figures()
if __name__ == '__main__':
    main()

