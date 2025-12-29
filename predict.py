from greedy import greedy_search_truncation_rank
from linear_prog import predict_ppl
import argparse
import torch
import copy

argparser = argparse.ArgumentParser()

argparser.add_argument(
    "--sen_list",
    type=str,
    default="",
    help="sensitivity list",
)
argparser.add_argument(
    "--succinct_calib",
    action="store_true",
    help="use succinct calibration",
)

def main(args):
    # target_ratios = [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6]
    target_ratios = [0.9]
   
    predicted_ppls = []
    
    sen_list = torch.load(args.sen_list)
    print("Original sensitivity list model.layers.11.self_attn.o_proj 0.1: ", sen_list["model.layers.11.self_attn.o_proj"][0.1])
    for target_ratio in target_ratios:
        # sen_list_copy = copy.deepcopy(sen_list)
        _, select_record = greedy_search_truncation_rank(raw_sensitivity_list=sen_list,
                                                        do_succinct_calib=args.succinct_calib,
                                                        param_ratio_target=target_ratio,
                                                        ratio_type="param_ratio")
        
        
        # Predict
        print("Before predict sensitivity list model.layers.11.self_attn.o_proj 0.1: ", sen_list["model.layers.11.self_attn.o_proj"][0.1])
        
        predicted_ppl, ppl_errors = predict_ppl(select_record=select_record, sen_list=sen_list)
        predicted_ppls.append(predicted_ppl)
        for name, ppl_error in ppl_errors.items():
            print("{}, {}, {}, {}".format(name, ppl_error[0], ppl_error[1], ppl_error[2]))

    predicted_ppls = [round(ppl, 2) for ppl in predicted_ppls]
    
    print("target ratios: ", target_ratios)
    print("predicted ppls: ", predicted_ppls)

if __name__ == '__main__':
   args = argparser.parse_args()
   main(args)
   
   
    
