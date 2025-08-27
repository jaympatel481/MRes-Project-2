import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from statsmodels.stats.proportion import proportions_ztest

# Set publication-quality parameters
plt.rcParams.update({
    'font.size': 12,
    'axes.linewidth': 1.5,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.minor.width': 1,
    'ytick.minor.width': 1,
    'font.family': 'Arial'
})

def load_and_process_data():
    """Load and process the annotation, non-sublineage datasets, and sublineage variants"""
    # Load full annotation data
    df_all = pd.read_csv("Final_annotated_variants_collated.csv")
    
    # Load non-sublineage variants
    df_nonsub = pd.read_csv("non_sublineage_variants_after_phasing_and_clustering.csv")
    
    # Load sublineage variants
    with open("sublineage_variants.txt") as f:
        sublineage_set = set(line.strip() for line in f if line.strip())
    
    # Parse variant_id for non-sublineage variants
    def parse_variant_id(vid):
        tokens = vid.split("_")
        chrom = tokens[0]
        pos = tokens[1]
        ref = tokens[2]
        alt = "_".join(tokens[3:]) if len(tokens) > 4 else tokens[3]
        return chrom, pos, ref, alt

    var_info = df_nonsub['variant_id'].apply(parse_variant_id)
    df_nonsub[['Chrom', 'Pos', 'Ref', 'Alt']] = pd.DataFrame(var_info.tolist(), index=df_nonsub.index)
    df_nonsub['Pos'] = df_nonsub['Pos'].astype(str)
    df_all['Pos'] = df_all['Pos'].astype(str)
    
    # Merge to get non-sublineage variants with annotations
    df_nonsub_annotated = pd.merge(
        df_nonsub,
        df_all,
        left_on=['sample', 'Chrom', 'Pos', 'Ref', 'Alt'],
        right_on=['Sample', 'Chrom', 'Pos', 'Ref', 'Alt'],
        how='left'
    )
    
    # Create variant_id for the full dataset to filter sublineage variants
    df_all['variant_id'] = (df_all['Chrom'] + '_' + 
                           df_all['Pos'].astype(str) + '_' + 
                           df_all['Ref'] + '_' + 
                           df_all['Alt'])
    
    # Create dataset with sublineage variants removed (but not considering phasing)
    df_no_direct_sublineage = df_all[~df_all['variant_id'].isin(sublineage_set)].copy()
    
    return df_all, df_nonsub_annotated, df_no_direct_sublineage

def classify_variants(df, genes):
    """Classify variants as non-synonymous or synonymous for each gene"""
    # Define effect categories
    nonsyn_effects = ['start_lost', 'stop_gained', 'stop_lost', 'missense_variant', 'start_gained', 'non_synonymous']
    syn_effects = ['synonymous_variant']
    
    results = []
    
    for sample in df['Sample'].unique():
        sample_data = df[df['Sample'] == sample]
        sample_result = {'Sample': sample}
        
        for gene in genes:
            if gene not in sample_data.columns:
                sample_result[f'{gene}_nonsyn'] = 0
                sample_result[f'{gene}_syn'] = 0
                continue
                
            gene_effects = sample_data[gene].fillna('').astype(str)
            
            # Count non-synonymous
            nonsyn_count = sum(any(effect in str(gene_effect).lower() for effect in nonsyn_effects) 
                             for gene_effect in gene_effects)
            
            # Count synonymous
            syn_count = sum(any(effect in str(gene_effect).lower() for effect in syn_effects) 
                          for gene_effect in gene_effects)
            
            sample_result[f'{gene}_nonsyn'] = 1 if nonsyn_count > 0 else 0
            sample_result[f'{gene}_syn'] = 1 if syn_count > 0 else 0
        
        results.append(sample_result)
    
    return pd.DataFrame(results)

def calculate_proportions(classified_df, genes):
    """Calculate proportions for each gene"""
    total_samples = len(classified_df)
    proportions = []
    
    for gene in genes:
        nonsyn_prop = classified_df[f'{gene}_nonsyn'].sum() / total_samples
        syn_prop = classified_df[f'{gene}_syn'].sum() / total_samples
        
        proportions.append({
            'Gene': gene,
            'Non-synonymous': nonsyn_prop,
            'Synonymous': syn_prop
        })
    
    return pd.DataFrame(proportions)

def add_proportions_test_pvalues(prop_df, classified_df, genes):
    """
    Add p-values for two-sample tests for proportions between non-synonymous and synonymous proportions for each gene.
    Returns updated DataFrame with an added 'pvalue' column.
    """
    total_samples = len(classified_df)
    pvalues = []

    for gene in genes:
        nonsyn_success = classified_df[f'{gene}_nonsyn'].sum()
        nonsyn_total = total_samples
        syn_success = classified_df[f'{gene}_syn'].sum()
        syn_total = total_samples

        successes = np.array([nonsyn_success, syn_success])
        samples = np.array([nonsyn_total, syn_total])

        try:
            stat, pval = proportions_ztest(successes, samples, alternative='two-sided')
        except Exception as e:
            pval = np.nan

        pvalues.append(pval)

    prop_df['pvalue'] = pvalues
    return prop_df

def create_proportion_plot(prop_df, title, ax):
    """Create a single proportion plot with p-value annotation"""
    genes = prop_df['Gene'].tolist()
    x_pos = np.arange(len(genes))
    
    # Colors - colorblind friendly
    nonsyn_color = '#D73027'  # Red
    syn_color = '#1A9850'     # Green
    
    width = 0.35
    
    # Create bars
    bars1 = ax.bar(x_pos - width/2, prop_df['Non-synonymous'], width, 
                   label='Non-synonymous', color=nonsyn_color, alpha=0.8, 
                   edgecolor='black', linewidth=1)
    
    bars2 = ax.bar(x_pos + width/2, prop_df['Synonymous'], width, 
                   label='Synonymous', color=syn_color, alpha=0.8, 
                   edgecolor='black', linewidth=1)
    
    # Customize plot
    ax.set_xlabel('Gene', fontsize=14, fontweight='bold')
    ax.set_ylabel('Proportion of Samples', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(genes, fontsize=12)
    ax.set_ylim(0, 1.1)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=10, 
                       fontweight='bold')
    
    # Add p-value annotation between bars for each gene
    if 'pvalue' in prop_df.columns:
        for i, pval in enumerate(prop_df['pvalue']):
            if not pd.isna(pval):
                max_height = max(prop_df.iloc[i]['Non-synonymous'], prop_df.iloc[i]['Synonymous'])
                # Format p-value appropriately
                if pval < 0.001:
                    pval_text = 'p<0.001'
                elif pval < 0.01:
                    pval_text = f'p={pval:.3f}'
                else:
                    pval_text = f'p={pval:.2f}'
                
                ax.text(x_pos[i], max_height + 0.08, pval_text, 
                       ha='center', va='bottom', fontsize=9, 
                       fontweight='normal', style='italic')
    
    # Customize spines and grid
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Legend
    legend = ax.legend(loc='upper right', frameon=True, fancybox=True, 
                      shadow=True, fontsize=12)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    return ax

def create_comparison_panel():
    """Create the complete three-panel comparison plot"""
    # Load and process data
    print("Loading and processing data...")
    df_all, df_nonsub_annotated, df_no_direct_sublineage = load_and_process_data()
    
    # Define genes
    genes = ['E1', 'E2', 'E4', 'E5', 'E6', 'E7', 'L1', 'L2', 'E1^E4']
    
    # Classify variants for all three datasets
    print("Classifying all variants...")
    classified_all = classify_variants(df_all, genes)
    
    print("Classifying variants without direct sublineage...")
    classified_no_direct_sub = classify_variants(df_no_direct_sublineage, genes)
    
    print("Classifying non-sublineage variants...")
    classified_nonsub = classify_variants(df_nonsub_annotated, genes)
    
    # Calculate proportions
    prop_all = calculate_proportions(classified_all, genes)
    prop_no_direct_sub = calculate_proportions(classified_no_direct_sub, genes)
    prop_nonsub = calculate_proportions(classified_nonsub, genes)
    
    # Add p-values for proportions tests
    print("Calculating proportions test p-values...")
    prop_all = add_proportions_test_pvalues(prop_all, classified_all, genes)
    prop_no_direct_sub = add_proportions_test_pvalues(prop_no_direct_sub, classified_no_direct_sub, genes)
    prop_nonsub = add_proportions_test_pvalues(prop_nonsub, classified_nonsub, genes)
    
    # Create the plot
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # Panel A: All variants
    create_proportion_plot(prop_all, 'A. All SNVs', axes[0])
    
    # Panel B: All variants except direct sublineage
    create_proportion_plot(prop_no_direct_sub, 'B. All SNVs (excl. sublineage)', axes[1])
    
    # Panel C: Non-sublineage variants
    create_proportion_plot(prop_nonsub, 'C. Non-sublineage SNVs', axes[2])
    
    # Overall title
    fig.suptitle('Proportion of Samples with Non-synonymous and Synonymous SNVs by Gene', 
                fontsize=18, fontweight='bold', y=0.95)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    
    # Save high-resolution figure
    plt.savefig('gene_mutation_proportions_three_panel.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig('gene_mutation_proportions_three_panel.pdf', bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    # Display summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    print("\nAll SNVs:")
    print(prop_all.round(3))
    print("\nAll SNVs (excluding direct sublineage):")
    print(prop_no_direct_sub.round(3))
    print("\nNon-sublineage SNVs:")
    print(prop_nonsub.round(3))
    
    # Save summary tables (including p-values)
    prop_all.to_csv('all_snvs_proportions_with_pvals.csv', index=False)
    prop_no_direct_sub.to_csv('no_direct_sublineage_snvs_proportions_with_pvals.csv', index=False)
    prop_nonsub.to_csv('nonsublineage_snvs_proportions_with_pvals.csv', index=False)
    
    print(f"\nPlots saved as:")
    print("- gene_mutation_proportions_three_panel.png")
    print("- gene_mutation_proportions_three_panel.pdf")
    print("- all_snvs_proportions_with_pvals.csv")
    print("- no_direct_sublineage_snvs_proportions_with_pvals.csv")
    print("- nonsublineage_snvs_proportions_with_pvals.csv")
    
    plt.show()
    
    return prop_all, prop_no_direct_sub, prop_nonsub

def create_individual_gene_plots():
    """Create individual plots for each gene with p-value annotations"""
    # Load and process data
    df_all, df_nonsub_annotated, df_no_direct_sublineage = load_and_process_data()
    genes = ['E1', 'E2', 'E4', 'E5', 'E6', 'E7', 'L1', 'L2', 'E1^E4']
    
    # Classify variants
    classified_all = classify_variants(df_all, genes)
    classified_no_direct_sub = classify_variants(df_no_direct_sublineage, genes)
    classified_nonsub = classify_variants(df_nonsub_annotated, genes)
    
    # Create individual plots for each gene
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()
    
    for i, gene in enumerate(genes):
        ax = axes[i]
        
        # Data for this gene
        data_all = [classified_all[f'{gene}_nonsyn'].sum() / len(classified_all),
                   classified_all[f'{gene}_syn'].sum() / len(classified_all)]
        data_no_direct_sub = [classified_no_direct_sub[f'{gene}_nonsyn'].sum() / len(classified_no_direct_sub),
                             classified_no_direct_sub[f'{gene}_syn'].sum() / len(classified_no_direct_sub)]
        data_nonsub = [classified_nonsub[f'{gene}_nonsyn'].sum() / len(classified_nonsub),
                      classified_nonsub[f'{gene}_syn'].sum() / len(classified_nonsub)]
        
        # Calculate p-values for this gene
        def get_pval(classified_df):
            successes = np.array([classified_df[f'{gene}_nonsyn'].sum(), classified_df[f'{gene}_syn'].sum()])
            samples = np.array([len(classified_df), len(classified_df)])
            try:
                _, pval = proportions_ztest(successes, samples, alternative='two-sided')
                return pval
            except:
                return np.nan
        
        pval_all = get_pval(classified_all)
        pval_no_direct_sub = get_pval(classified_no_direct_sub)
        pval_nonsub = get_pval(classified_nonsub)
        
        x_pos = np.arange(2)
        width = 0.25
        
        bars1 = ax.bar(x_pos - width, data_all, width, label='All SNVs', 
                      color='#377eb8', alpha=0.8, edgecolor='black')
        bars2 = ax.bar(x_pos, data_no_direct_sub, width, label='Excl. sublineage', 
                      color='#4daf4a', alpha=0.8, edgecolor='black')
        bars3 = ax.bar(x_pos + width, data_nonsub, width, label='Non-sublineage', 
                      color='#e41a1c', alpha=0.8, edgecolor='black')
        
        ax.set_title(f'{gene}', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(['Non-synonymous', 'Synonymous'], fontsize=10)
        ax.set_ylabel('Proportion', fontsize=12)
        ax.set_ylim(0, 1.1)
        
        # Add value labels
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                           f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        # Add p-value annotations
        pvals = [pval_all, pval_no_direct_sub, pval_nonsub]
        colors = ['#377eb8', '#4daf4a', '#e41a1c']
        x_positions = [0.5, 1.5, 2.5]
        
        for j, (pval, color) in enumerate(zip(pvals, colors)):
            if not pd.isna(pval):
                pval_text = 'p<0.001' if pval < 0.001 else f'p={pval:.3f}'
                max_height = max([data_all, data_no_direct_sub, data_nonsub][j])
                ax.text(j - 0.5, max_height + 0.08, pval_text, ha='center', va='bottom', 
                       fontsize=7, style='italic', color=color)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        if i == 0:  # Add legend to first plot
            ax.legend(loc='upper right', fontsize=9)
    
    plt.suptitle('Gene-specific Mutation Proportions: All vs Excl. Sublineage vs Non-sublineage SNVs\n(p-values show tests between non-synonymous and synonymous proportions)', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('individual_gene_proportions_three_categories_with_pvals.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Create main comparison panel
    prop_all, prop_no_direct_sub, prop_nonsub = create_comparison_panel()
    
    # Optionally create individual gene plots
    print("\nCreating individual gene plots...")
    create_individual_gene_plots()
    
    print("\nAnalysis complete!")

