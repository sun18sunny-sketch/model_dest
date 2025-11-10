import argparse
import os
import os.path
import sys
import pandas as pd
import settings
settings.pickle_path = './'
import call_llm
parser = argparse.ArgumentParser(description='Score')
parser.add_argument('-m', dest='modelFile', help='model filename, default will be the model file hardcoded in the script')
parser.add_argument('-i', dest='scoreInputCSV', required=True, help='input filename')
parser.add_argument('-o', dest='scoreOutputCSV', required=True, help='output csv filename')
args = parser.parse_args()
model_file = args.modelFile
score_input_csv = args.scoreInputCSV
score_output_csv = args.scoreOutputCSV
if not os.path.isfile(score_input_csv):
    print('Not found input file',score_input_csv)
    sys.exit()
input_df = pd.read_csv(os.path.join(os.getcwd(), score_input_csv))
rowResult = []
for i, row in input_df.iterrows():
    rowResult.append(call_llm.scoreModel(row['userPrompt'],row['API_KEY']))
output_df = pd.DataFrame(rowResult, columns=['response', 'llmMetrics'])
output_df = pd.merge(input_df, output_df, how='inner', left_index=True, right_index=True)
output_df.to_csv(score_output_csv, sep=',', index=False)
