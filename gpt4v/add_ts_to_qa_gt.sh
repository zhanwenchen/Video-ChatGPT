export SPLIT=train # train, val, test
python /home/zhanwen/vtom/gpt4v/add_ts_to_qa_gt.py --gpt4v_result_dirpath /home/zhanwen/vtom/gpt4v/result_${SPLIT} --qa_file_in /home/zhanwen/vtom/data/siq2/qa/qa_${SPLIT}.json --qa_file_out /home/zhanwen/vtom/data/siq2/qa/qa_${SPLIT}_with_ts.json
export SPLIT=val # train, val, test
python /home/zhanwen/vtom/gpt4v/add_ts_to_qa_gt.py --gpt4v_result_dirpath /home/zhanwen/vtom/gpt4v/result_${SPLIT} --qa_file_in /home/zhanwen/vtom/data/siq2/qa/qa_${SPLIT}.json --qa_file_out /home/zhanwen/vtom/data/siq2/qa/qa_${SPLIT}_with_ts.json
export SPLIT=test # train, val, test
python /home/zhanwen/vtom/gpt4v/add_ts_to_qa_gt.py --gpt4v_result_dirpath /home/zhanwen/vtom/gpt4v/result_${SPLIT} --qa_file_in /home/zhanwen/vtom/data/siq2/qa/qa_${SPLIT}.json --qa_file_out /home/zhanwen/vtom/data/siq2/qa/qa_${SPLIT}_with_ts.json
