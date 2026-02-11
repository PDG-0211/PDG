import logging
import linecache
import subprocess
import torch
from torch.utils.data.dataset import Dataset

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    def __init__(self, file_name, tokenizer, max_input_len, max_output_len):
        self.file_name = file_name
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.tokenizer = tokenizer
        self.total_size = int(subprocess.check_output(
            "wc -l " + file_name, shell=True).split()[0])
    def __getitem__(self, index):
        line = linecache.getline(self.file_name, index + 1)
        method = eval(line)

        input_txt = self.get_input_txt(method)
        input_ids = self.tokenizer.encode(input_txt, return_tensors="pt", 
                                          max_length=self.max_input_len, padding="max_length", truncation=True).squeeze(0)
        input_attention_mask = torch.zeros_like(input_ids)
        input_attention_mask[input_ids != self.tokenizer.pad_token_id] = 1

        output_txt = self.get_output_txt(method)
        output_ids = self.tokenizer.encode(output_txt, return_tensors="pt",
                                           max_length=self.max_output_len, padding="max_length", truncation=True).squeeze(0)
        output_ids[output_ids == self.tokenizer.pad_token_id] = -100     # -100是CrossEntropyLoss的ignore_index
        
        result = {
            "input_ids": input_ids,
            "input_attention_mask": input_attention_mask,
            "output_ids": output_ids
        }
        return result
        
    def __len__(self):
        return self.total_size
    
    def get_input_txt(self, method) -> str:
        code_txt = method["code"]
        param_name = method["param_name"]
        input_txt = param_name + self.tokenizer.sep_token + code_txt
        return input_txt
    
    def get_output_txt(self, method):
        return method["param_comment"]
    
class CodeSliceDataset(Dataset):
    def __init__(self, file_name, tokenizer, max_input_len, max_output_len):
        self.file_name = file_name
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.tokenizer = tokenizer
        self.total_size = int(subprocess.check_output(
            "wc -l " + file_name, shell=True).split()[0])
    def __getitem__(self, index):
        line = linecache.getline(self.file_name, index + 1)
        method = eval(line)

        input_txt = self.get_input_txt(method)
        input_ids = self.tokenizer.encode(input_txt, return_tensors="pt", 
                                          max_length=self.max_input_len, padding="max_length", truncation=True).squeeze(0)
        input_attention_mask = torch.zeros_like(input_ids)
        input_attention_mask[input_ids != self.tokenizer.pad_token_id] = 1

        output_txt = self.get_output_txt(method)
        output_ids = self.tokenizer.encode(output_txt, return_tensors="pt",
                                           max_length=self.max_output_len, padding="max_length", truncation=True).squeeze(0)
        output_ids[output_ids == self.tokenizer.pad_token_id] = -100     # -100是CrossEntropyLoss的ignore_index
        
        result = {
            "input_ids": input_ids,
            "input_attention_mask": input_attention_mask,
            "output_ids": output_ids
        }
        return result
        
    def __len__(self):
        return self.total_size
    
    def get_input_txt(self, method) -> str:
        code_txt = ""
        for code_slice in method["code_slice"]:
            code_txt += code_slice
        param_name = method["param_name"]
        input_txt = param_name + self.tokenizer.sep_token + code_txt
        return input_txt
    
    def get_output_txt(self, method):
        return method["param_comment"]
    
class ASTDataset(Dataset):
    def __init__(self, file_name, tokenizer, max_input_len, max_output_len):
        self.file_name = file_name
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.tokenizer = tokenizer
        self.total_size = int(subprocess.check_output(
            "wc -l " + file_name, shell=True).split()[0])
    def __getitem__(self, index):
        line = linecache.getline(self.file_name, index + 1)
        method = eval(line)

        input_txt = self.get_input_txt(method)
        input_ids = self.tokenizer.encode(input_txt, return_tensors="pt", 
                                          max_length=self.max_input_len, padding="max_length", truncation=True).squeeze(0)
        input_attention_mask = torch.zeros_like(input_ids)
        input_attention_mask[input_ids != self.tokenizer.pad_token_id] = 1

        output_txt = self.get_output_txt(method)
        output_ids = self.tokenizer.encode(output_txt, return_tensors="pt",
                                           max_length=self.max_output_len, padding="max_length", truncation=True).squeeze(0)
        output_ids[output_ids == self.tokenizer.pad_token_id] = -100     # -100是CrossEntropyLoss的ignore_index
        
        result = {
            "input_ids": input_ids,
            "input_attention_mask": input_attention_mask,
            "output_ids": output_ids
        }
        return result
        
    def __len__(self):
        return self.total_size
    
    def get_input_txt(self, method) -> str:
        ast_txt = " ".join(method["ast"])

        param_name = method["param_name"]
        input_txt = param_name + self.tokenizer.sep_token + ast_txt
        return input_txt
    
    def get_output_txt(self, method):
        return method["param_comment"]


class ParamDesDataset(Dataset):
    def __init__(self, file_name, tokenizer, max_input_len, max_output_len):
        self.file_name = file_name
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.tokenizer = tokenizer
        self.total_size = int(subprocess.check_output(
            "wc -l " + file_name, shell=True).split()[0])
    def __getitem__(self, index):
        line = linecache.getline(self.file_name, index + 1)
        method = eval(line)
    
        input_txt = self.get_input_txt(method)
        input_ids = self.tokenizer.encode(input_txt, return_tensors="pt", 
                                          max_length=self.max_input_len, padding="max_length", truncation=True).squeeze(0)
        input_attention_mask = torch.zeros_like(input_ids)
        input_attention_mask[input_ids != self.tokenizer.pad_token_id] = 1

        output_txt = self.get_output_txt(method)
        output_ids = self.tokenizer.encode(output_txt, return_tensors="pt",
                                           max_length=self.max_output_len, padding="max_length", truncation=True).squeeze(0)
        output_ids[output_ids == self.tokenizer.pad_token_id] = -100     # -100是CrossEntropyLoss的ignore_index
        
        result = {
            "input_ids": input_ids,
            "input_attention_mask": input_attention_mask,
            "output_ids": output_ids
        }
        return result
        
    def __len__(self):
        return self.total_size
    
    def get_input_txt(self, method) -> str:
        code_txt = ""
        for code_slice in method["code_slice"]:
            code_txt += code_slice
        param_flow = ' '.join(method["param_flow"])
        param_name = method["param_name"]
        input_txt = code_txt + self.tokenizer.sep_token + param_flow + self.tokenizer.sep_token + param_name
        return input_txt
    
    def get_output_txt(self, method):
        return method["param_comment"]

class CustomizeDataset1(Dataset):
    def __init__(self, file_name, tokenizer, max_input_len, max_output_len):
        self.file_name = file_name
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.tokenizer = tokenizer
        self.total_size = int(subprocess.check_output(
            "wc -l " + file_name, shell=True).split()[0])
    def __getitem__(self, index):
        line = linecache.getline(self.file_name, index + 1)
        method = eval(line)

        input_txt = self.get_input_txt(method)
        input_ids = self.tokenizer.encode(input_txt, return_tensors="pt", 
                                          max_length=self.max_input_len, padding="max_length", truncation=True).squeeze(0)
        input_attention_mask = torch.zeros_like(input_ids)
        input_attention_mask[input_ids != self.tokenizer.pad_token_id] = 1

        output_txt = self.get_output_txt(method)
        output_ids = self.tokenizer.encode(output_txt, return_tensors="pt",
                                           max_length=self.max_output_len, padding="max_length", truncation=True).squeeze(0)
        output_ids[output_ids == self.tokenizer.pad_token_id] = -100     # -100是CrossEntropyLoss的ignore_index
        
        result = {
            "input_ids": input_ids,
            "input_attention_mask": input_attention_mask,
            "output_ids": output_ids
        }
        return result
        
    def __len__(self):
        return self.total_size
    
    def get_input_txt(self, method) -> str:
        code_txt = method["code"]
        param_flow = ' '.join(method["param_flow"])
        param_name = method["param_name"]
        input_txt = param_name + self.tokenizer.sep_token + param_flow + self.tokenizer.sep_token + code_txt
        return input_txt
    
    def get_output_txt(self, method):
        return method["param_comment"]
    

class T5EncoderOnlyDataset(Dataset):
    def __init__(self, file_name, tokenizer, max_input_len, max_output_len):
        self.file_name = file_name
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.tokenizer = tokenizer
        self.total_size = int(subprocess.check_output(
            "wc -l " + file_name, shell=True).split()[0])
    def __getitem__(self, index):
        line = linecache.getline(self.file_name, index + 1)
        method = eval(line)

        source_txt = self.get_input_txt(method)
        source_ids = self.tokenizer.encode(source_txt, return_tensors="pt", 
                                          max_length=self.max_input_len, padding="max_length", truncation=True).squeeze(0)
        source_mask = torch.zeros_like(source_ids)
        source_mask[source_ids != self.tokenizer.pad_token_id] = 1

        target_txt = self.get_output_txt(method)
        target_ids = self.tokenizer.encode(target_txt, return_tensors="pt",
                                           max_length=self.max_output_len, padding="max_length", truncation=True).squeeze(0)
        target_mask = torch.zeros_like(target_ids)
        target_mask[target_ids != self.tokenizer.pad_token_id] = 1
        
        result = {
            "source_ids": source_ids,
            "source_mask": source_mask,
            "target_ids": target_ids,
            "target_mask": target_mask
        }
        return result
        
    def __len__(self):
        return self.total_size
    
    def get_input_txt(self, method) -> str:
        code_txt = method["code"]
        param_name = method["param_name"]
        input_txt = param_name + self.tokenizer.sep_token + code_txt
        return input_txt
    
    def get_output_txt(self, method):
        return method["param_comment"]
    
class MultiEncoderDataset(Dataset):
    def __init__(self, file_name, tokenizer, max_input_len, max_output_len):
        self.file_name = file_name
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.tokenizer = tokenizer
        self.total_size = int(subprocess.check_output(
            "wc -l " + file_name, shell=True).split()[0])
    def __getitem__(self, index):
        line = linecache.getline(self.file_name, index + 1)
        method = eval(line)

        input_txt1 = self.get_input_txt1(method)
        input_ids1 = self.tokenizer.encode(input_txt1, return_tensors="pt", 
                                          max_length=self.max_input_len, padding="max_length", truncation=True).squeeze(0)
        input_attention_mask1 = torch.zeros_like(input_ids1)
        input_attention_mask1[input_ids1 != self.tokenizer.pad_token_id] = 1

        input_txt2 = self.get_input_txt2(method)
        input_ids2 = self.tokenizer.encode(input_txt2, return_tensors="pt", 
                                          max_length=self.max_input_len, padding="max_length", truncation=True).squeeze(0)
        input_attention_mask2 = torch.zeros_like(input_ids2)
        input_attention_mask2[input_ids2 != self.tokenizer.pad_token_id] = 1

        output_txt = self.get_output_txt(method)
        output_ids = self.tokenizer.encode(output_txt, return_tensors="pt",
                                           max_length=self.max_output_len, padding="max_length", truncation=True).squeeze(0)
        output_ids[output_ids == self.tokenizer.pad_token_id] = -100     # -100是CrossEntropyLoss的ignore_index
        
        result = {
            "input_ids1": input_ids1,
            "input_attention_mask1": input_attention_mask1,
            "input_ids2": input_ids2,
            "input_attention_mask2": input_attention_mask2,
            "output_ids": output_ids
        }
        return result
        
    def __len__(self):
        return self.total_size
    
    def get_input_txt1(self, method) -> str:
        code_txt = method["code"]
        input_txt = code_txt
        return input_txt
    
    def get_input_txt2(self, method) -> str:
        param_flow = ' '.join(method["param_flow"])
        param_name = method["param_name"]
        input_txt = param_name + self.tokenizer.sep_token + param_flow
        return input_txt
    
    def get_output_txt(self, method):
        return method["param_comment"]
    
    
class MultiEncoderASTDataset(Dataset):
    def __init__(self, file_name, tokenizer, max_input_len, max_output_len):
        self.file_name = file_name
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.tokenizer = tokenizer
        self.total_size = int(subprocess.check_output(
            "wc -l " + file_name, shell=True).split()[0])
    def __getitem__(self, index):
        line = linecache.getline(self.file_name, index + 1)
        method = eval(line)

        input_txt1 = self.get_input_txt1(method)
        input_ids1 = self.tokenizer.encode(input_txt1, return_tensors="pt", 
                                          max_length=self.max_input_len, padding="max_length", truncation=True).squeeze(0)
        input_attention_mask1 = torch.zeros_like(input_ids1)
        input_attention_mask1[input_ids1 != self.tokenizer.pad_token_id] = 1

        input_txt2 = self.get_input_txt2(method)
        input_ids2 = self.tokenizer.encode(input_txt2, return_tensors="pt", 
                                          max_length=self.max_input_len, padding="max_length", truncation=True).squeeze(0)
        input_attention_mask2 = torch.zeros_like(input_ids2)
        input_attention_mask2[input_ids2 != self.tokenizer.pad_token_id] = 1
 
        output_txt = self.get_output_txt(method)
        output_ids = self.tokenizer.encode(output_txt, return_tensors="pt",
                                           max_length=self.max_output_len, padding="max_length", truncation=True).squeeze(0)
        output_ids[output_ids == self.tokenizer.pad_token_id] = -100     # -100是CrossEntropyLoss的ignore_index
        
        result = {
            "input_ids1": input_ids1,
            "input_attention_mask1": input_attention_mask1,
            "input_ids2": input_ids2,
            "input_attention_mask2": input_attention_mask2,
            "output_ids": output_ids
        }
        return result
        
    def __len__(self):
        return self.total_size
    
    def get_input_txt1(self, method) -> str:
        code_txt = method["code"]
        param_name = method["param_name"]
        input_txt = param_name + self.tokenizer.sep_token+code_txt
        return input_txt
    
    def get_input_txt2(self, method) -> str:
        ast = ' '.join(method["ast"])
        param_name = method["param_name"]
        input_txt =  param_name + self.tokenizer.sep_token + ast
        return input_txt
    
    def get_output_txt(self, method):
        return method["param_comment"]

class MultiEncoderCodeSliceASTDataset(Dataset):
    def __init__(self, file_name, tokenizer, max_input_len, max_output_len):
        self.file_name = file_name
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.tokenizer = tokenizer
        self.total_size = int(subprocess.check_output(
            "wc -l " + file_name, shell=True).split()[0])
    def __getitem__(self, index):
        line = linecache.getline(self.file_name, index + 1)
        method = eval(line)

        input_txt1 = self.get_input_txt1(method)
        input_ids1 = self.tokenizer.encode(input_txt1, return_tensors="pt", 
                                          max_length=self.max_input_len, padding="max_length", truncation=True).squeeze(0)
        input_attention_mask1 = torch.zeros_like(input_ids1)
        input_attention_mask1[input_ids1 != self.tokenizer.pad_token_id] = 1

        input_txt2 = self.get_input_txt2(method)
        input_ids2 = self.tokenizer.encode(input_txt2, return_tensors="pt", 
                                          max_length=self.max_input_len, padding="max_length", truncation=True).squeeze(0)
        input_attention_mask2 = torch.zeros_like(input_ids2)
        input_attention_mask2[input_ids2 != self.tokenizer.pad_token_id] = 1

        output_txt = self.get_output_txt(method)
        output_ids = self.tokenizer.encode(output_txt, return_tensors="pt",
                                           max_length=self.max_output_len, padding="max_length", truncation=True).squeeze(0)
        output_ids[output_ids == self.tokenizer.pad_token_id] = -100     # -100是CrossEntropyLoss的ignore_index
        
        result = {
            "input_ids1": input_ids1,
            "input_attention_mask1": input_attention_mask1,
            "input_ids2": input_ids2,
            "input_attention_mask2": input_attention_mask2,
            "output_ids": output_ids
        }
        return result
        
    def __len__(self):
        return self.total_size
    
    def get_input_txt1(self, method) -> str:
        code_txt = ''.join(method["code_slice"])
        param_name = method["param_name"]
        input_txt = param_name + self.tokenizer.sep_token+code_txt
        return input_txt
    
    def get_input_txt2(self, method) -> str:
        ast = ' '.join(method["ast"])
        param_name = method["param_name"]
        input_txt =  param_name + self.tokenizer.sep_token + ast
        return input_txt
    
    def get_output_txt(self, method):
        return method["param_comment"]

class MultiEncoderCodeCodeSliceDataset(Dataset):
    def __init__(self, file_name, tokenizer, max_input_len, max_output_len):
        self.file_name = file_name
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.tokenizer = tokenizer
        self.total_size = int(subprocess.check_output(
            "wc -l " + file_name, shell=True).split()[0])
    def __getitem__(self, index):
        line = linecache.getline(self.file_name, index + 1)
        method = eval(line)

        input_txt1 = self.get_input_txt1(method)
        input_ids1 = self.tokenizer.encode(input_txt1, return_tensors="pt", 
                                          max_length=self.max_input_len, padding="max_length", truncation=True).squeeze(0)
        input_attention_mask1 = torch.zeros_like(input_ids1)
        input_attention_mask1[input_ids1 != self.tokenizer.pad_token_id] = 1

        input_txt2 = self.get_input_txt2(method)
        input_ids2 = self.tokenizer.encode(input_txt2, return_tensors="pt", 
                                          max_length=self.max_input_len, padding="max_length", truncation=True).squeeze(0)
        input_attention_mask2 = torch.zeros_like(input_ids2)
        input_attention_mask2[input_ids2 != self.tokenizer.pad_token_id] = 1

        output_txt = self.get_output_txt(method)
        output_ids = self.tokenizer.encode(output_txt, return_tensors="pt",
                                           max_length=self.max_output_len, padding="max_length", truncation=True).squeeze(0)
        output_ids[output_ids == self.tokenizer.pad_token_id] = -100     # -100是CrossEntropyLoss的ignore_index
        
        result = {
            "input_ids1": input_ids1,
            "input_attention_mask1": input_attention_mask1,
            "input_ids2": input_ids2,
            "input_attention_mask2": input_attention_mask2,
            "output_ids": output_ids
        }
        return result
        
    def __len__(self):
        return self.total_size
    
    def get_input_txt1(self, method) -> str:
        code_slice = ' '.join(method["code_slice"])
        param_name = method["param_name"]
        input_txt =  param_name + self.tokenizer.sep_token+code_slice
        return input_txt
    def get_input_txt2(self, method) -> str:
        code_txt = method["code"]
        input_txt = code_txt
        return input_txt
    
    def get_output_txt(self, method):
        return method["param_comment"]