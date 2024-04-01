from datasets import load_dataset
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

ds = load_dataset("naorm/caption-eval")
ds2 = load_dataset("naorm/caption-eval-blip2")
ds3 = load_dataset("naorm/caption-eval-blip2-quant")

df = pd.concat([ds['train'].to_pandas(), ds2['train'].to_pandas(), ds3['train'].to_pandas()]).reset_index(drop=True)

plt.figure(figsize=(30,20))
sns.barplot(data=df[(df['metric_name']!='sacrebleu')&(~df['model_name'].str.contains('bit'))], x='model_name', y='value', hue='metric_name')
plt.savefig("reports/figures/eval.png", dpi=400, bbox_inches="tight")

plt.figure(figsize=(20,10))
sns.barplot(data=df[(df['metric_name']=='sacrebleu')&(~df['model_name'].str.contains('bit'))], x='model_name', y='value', hue='metric_name')
plt.savefig("reports/figures/sacrebleu.png", dpi=400, bbox_inches="tight")

plt.figure(figsize=(30,20))
sns.barplot(data=df[(df['metric_name']!='sacrebleu')&(df['model_name'].str.contains('blip2'))], x='model_name', y='value', hue='metric_name')
plt.savefig("reports/figures/quant_eval.png", dpi=400, bbox_inches="tight")

plt.figure(figsize=(20,10))
sns.barplot(data=df[(df['metric_name']=='sacrebleu')&(df['model_name'].str.contains('blip2'))], x='model_name', y='value', hue='metric_name')
plt.savefig("reports/figures/quant_sacrebleu.png", dpi=400, bbox_inches="tight")