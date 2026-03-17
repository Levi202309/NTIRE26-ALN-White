# Official Implementation of NTIRE26 ALN-White Competition 


1.Download the pre-trained weights at: [https://drive.google.com/file/d/1U2AdyKjiFHq0nHfqBWKAKhU2LnF0xO9S/view?usp=sharing](https://drive.google.com/file/d/1U2AdyKjiFHq0nHfqBWKAKhU2LnF0xO9S/view?usp=sharing)

2.Install the dependencies from `requirements.txt`:

`pip install -r requirements.txt`
  
3.Modify the paths in `inference.sh` to match the directories on your own machine.

Modify the path to your `testing_ALNwhite_mamba_jrx.py` script.

Use `--experiment_name` to specify the generated folder, 

`--pre_model` for the weight path, 

`--unified_path` for the root directory of the generated path, 

and `--eval_in_path` for the test_in path

4.Run the inference script:

`bash inference.sh`




