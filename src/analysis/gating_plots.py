# src/analysis/gating_plots.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ---------------------------------------------------------
# 1. DATA INPUT (Extracted from Logs)
# ---------------------------------------------------------
all_data = []

# --- URBAN DATASETS (Expectation: Spatial > Spectral, g > 0.5) ---
houston_data = [
    ("Soil", 0.5884), ("Parking Lot 2", 0.6011), ("Parking Lot 1", 0.6263),
    ("Tennis Court", 0.5972), ("Stressed grass", 0.5643), ("Road", 0.6284),
    ("Railway", 0.6291), ("Highway", 0.6072), ("Trees", 0.6215),
    ("Synthetic grass", 0.5904), ("Residential", 0.6128), ("Undefined", 0.6033),
    ("Commercial", 0.6041), ("Healthy grass", 0.6098), ("Water", 0.5831)
]
for c, g in houston_data: all_data.append({"Dataset": "Houston13", "Domain": "Urban", "Class": c, "Gate": g})

pavia_u_data = [
    ("Self-Blocking Bricks", 0.4981), ("Bare Soil", 0.5099), ("Trees", 0.5068),
    ("Meadows", 0.4776), ("Gravel", 0.5028), ("Bitumen", 0.5027),
    ("Painted metal sheets", 0.4994), ("Undefined", 0.5123), ("Asphalt", 0.4949)
]
for c, g in pavia_u_data: all_data.append({"Dataset": "Pavia Univ.", "Domain": "Urban", "Class": c, "Gate": g})

pavia_c_data = [
    ("Asphalt", 0.5209), ("Meadows", 0.5150), ("Trees", 0.5065),
    ("Self-Blocking Bricks", 0.4927), ("Tiles", 0.5605), ("Water", 0.5045),
    ("Bitumen", 0.5257), ("Shadows", 0.5301), ("Undefined", 0.5316)
]
for c, g in pavia_c_data: all_data.append({"Dataset": "Pavia Centre", "Domain": "Urban", "Class": c, "Gate": g})

# --- AGRICULTURE DATASETS (Expectation: Balanced or Spectral Bias, g ~ 0.5) ---
indian_pines_data = [
    ("Hay-windrowed", 0.4864), ("Grass-trees", 0.5079), ("Undefined", 0.5152),
    ("Buildings-grass...", 0.5208), ("Soybean-clean", 0.5022), ("Corn-mintill", 0.4727),
    ("Woods", 0.4888), ("Grass-pasture-mowed", 0.5065), ("Corn", 0.5069),
    ("Soybean-mintill", 0.4947), ("Grass-Pasture", 0.5051), ("Corn-notill", 0.5071),
    ("Oats", 0.5080), ("Wheat", 0.5017), ("Alfalfa", 0.4878), ("Soybean-notill", 0.5081)
]
for c, g in indian_pines_data: all_data.append({"Dataset": "Indian Pines", "Domain": "Agriculture", "Class": c, "Gate": g})

salinas_data = [
    ("Lettuce_romaine_5wk", 0.5291), ("Corn_senesced...", 0.5022), ("Lettuce_romaine_6wk", 0.5451),
    ("Fallow", 0.4939), ("Vineyard_untrained", 0.5066), ("Lettuce_romaine_4wk", 0.5329),
    ("Broccoli_green_2", 0.5042), ("Undefined", 0.4853), ("Fallow_rough_plow", 0.4605),
    ("Soil_vineyard_develop", 0.5109), ("Stubble", 0.5530), ("Broccoli_green_1", 0.4879),
    ("Fallow_smooth", 0.4781), ("Grapes_untrained", 0.4947), ("Lettuce_romaine_7wk", 0.5732),
    ("Celery", 0.5800)
]
for c, g in salinas_data: all_data.append({"Dataset": "Salinas", "Domain": "Agriculture", "Class": c, "Gate": g})

salinas_a_data = [
    ("Undefined", 0.5019), ("Corn_senesced...", 0.4421), ("Lettuce_romaine_5wk", 0.4744),
    ("Lettuce_romaine_6wk", 0.5404), ("Broccoli_green_1", 0.5297), ("Lettuce_romaine_4wk", 0.5246)
]
for c, g in salinas_a_data: all_data.append({"Dataset": "SalinasA", "Domain": "Agriculture", "Class": c, "Gate": g})

botswana_data = [
    ("Mixed mopane", 0.5226), ("Water", 0.5106), ("Acacia grasslands", 0.5165),
    ("Firescar", 0.5056), ("Floodplain grasses 1", 0.4952), ("Acacia woodlands", 0.5265),
    ("Hippo grass", 0.5016), ("Riparian", 0.5045), ("Short mopane", 0.5204),
    ("Floodplain grasses 2", 0.5017), ("Reeds", 0.5001), ("Undefined", 0.5126),
    ("Acacia shrublands", 0.5201), ("Island interior", 0.5051)
]
for c, g in botswana_data: all_data.append({"Dataset": "Botswana", "Domain": "Nature/Mixed", "Class": c, "Gate": g})

ksc_data = [
    ("Hardwood", 0.4958), ("Slash pine", 0.5211), ("Oak/broadleaf", 0.5217),
    ("Scrub", 0.5249), ("Cabbage palm", 0.5173), ("Willow swamp", 0.5061),
    ("Spartina marsh", 0.5145), ("Cattail marsh", 0.4915), ("Swamp", 0.5077),
    ("Salt marsh", 0.4826), ("Graminoid marsh", 0.4703), ("Undefined", 0.5069),
    ("Mud flats", 0.5057)
]
for c, g in ksc_data: all_data.append({"Dataset": "Kennedy SC", "Domain": "Nature/Mixed", "Class": c, "Gate": g})

# Create DataFrame
df = pd.DataFrame(all_data)

# ---------------------------------------------------------
# 2. GRAPH SETTINGS 
# ---------------------------------------------------------
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 11
plt.figure(figsize=(14, 8), dpi=300)

# Color Palette: Urban=Reddish, Agri=Greenish, Nature=Blueish
palette = {"Urban": "#e74c3c", "Agriculture": "#27ae60", "Nature/Mixed": "#3498db"}

# ---------------------------------------------------------
# 3. PLOTTING: Box Plot + Strip Plot
# ---------------------------------------------------------
# Order: Urban -> Agri -> Nature
order = ["Houston13", "Pavia Centre", "Pavia Univ.", "Salinas", "SalinasA", "Indian Pines", "Botswana", "Kennedy SC"]

# Box Plot (Distribution body)
sns.boxplot(x='Dataset', y='Gate', hue='Domain', data=df, order=order,
            dodge=False, width=0.6, palette=palette, linewidth=1.5, fliersize=0)

# Strip Plot (Individual Classes)
sns.stripplot(x='Dataset', y='Gate', data=df, order=order,
              color='black', size=4, alpha=0.6, jitter=0.2)

# Reference Line (0.5 Balanced)
plt.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.8)
plt.text(0.5, 0.495, 'Balanced Fusion (g=0.5)', color='gray', fontsize=9, ha='right')

# ---------------------------------------------------------
# 4. ANNOTATIONS (Key Findings)
# ---------------------------------------------------------
# Houston - High Spatial Bias
plt.annotate('Strong Spatial Bias\n(Urban Structures)', xy=(0, 0.63), xytext=(0.5, 0.64),
             arrowprops=dict(arrowstyle="->", color='black', connectionstyle="arc3,rad=.2"),
             fontsize=9, color='#c0392b', ha='center')

# Salinas - Wide Range
plt.annotate('High Variance\n(Diverse Crop Types)', xy=(3, 0.58), xytext=(3, 0.60),
             arrowprops=dict(arrowstyle="->", color='black'),
             fontsize=9, color='#27ae60', ha='center')

# SalinasA - Lowest Spectral
plt.annotate('Spectral Dominance\n(Senesced Crops)', xy=(4, 0.442), xytext=(4, 0.42),
             arrowprops=dict(arrowstyle="->", color='black'),
             fontsize=9, color='#27ae60', ha='center')

# ---------------------------------------------------------
# 5. ADJUSTMENTS AND SAVING
# ---------------------------------------------------------
plt.title('Analysis of Learned Gating Parameters Across 8 Hyperspectral Datasets', fontsize=16, pad=20)
plt.ylabel('Gating Parameter (g)\n(>0.5 Spatial Bias | <0.5 Spectral Bias)', fontsize=12)
plt.xlabel('')
plt.ylim(0.41, 0.66)
plt.legend(loc='upper right', title='Scene Domain')
sns.despine(trim=True)
plt.grid(axis='y', linestyle=':', alpha=0.4)

plt.tight_layout()
plt.savefig('comprehensive_gating_analysis.png', dpi=300, bbox_inches='tight')
plt.savefig('comprehensive_gating_analysis.eps', dpi=300, bbox_inches='tight')
plt.show()

print("Graph successfully created: comprehensive_gating_analysis.png and comprehensive_gating_analysis.eps")