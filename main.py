import torch
import transformers
import util
import json
import pandas as pd
import os
import argparse
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score,precision_score,recall_score
from tqdm import trange
from torch.utils.data import Subset

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description='Set training parameters')
    parser.add_argument('--task', type=str, default='AG', help='Task name')
    parser.add_argument('--embedding_tool', type=str, default='openai', help='Embedding tool')
    parser.add_argument('--compression_style', type=str, default='autoencoder', help='Dimension compression method')
    parser.add_argument('--is_ct', type=str, default='true', help='Whether to import counterfactual samples: true or false?')
    parser.add_argument('--slm', type=str, default='', help='SLM path')
    parser.add_argument('--emb_prompt', type=str, default='Summarize this entity record: ', help='Embedding prompt')
    parser.add_argument('--ct_prompt', type=str, default='normal', help='CT prompt type')
    parser.add_argument('--ct_llm', type=str, default='gpt-4o-mini', help='CT LLM')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden layer dimension')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epoch_count_for_valid', type=int, default=1, help='Number of epochs between evaluations')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--alpha', type=float, default=1.0, help='Alignment loss weight')
    parser.add_argument('--beta', type=float, default=1.0, help='Reconstruction loss weight')
    parser.add_argument('--lamda', type=float, default=1.0, help='Reconstruction loss difference enhancement')
    parser.add_argument('--train_data_ratio', type=float, default=1.0, help='Training data ratio')
    parser.add_argument('--wrong_label_ratio', type=float, default=0.0, help='Wrong label ratio')
    parser.add_argument('--ct_ratio', type=float, default=0.5, help='Counterfactual sample ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    print(args)

    task = args.task
    embedding_tool = args.embedding_tool
    slm = args.slm
    emb_prompt = args.emb_prompt
    num_epochs = args.num_epochs
    lr = args.lr
    hidden_dim = args.hidden_dim
    batch_size = args.batch_size
    epoch_count_for_valid = args.epoch_count_for_valid 
    alpha = args.alpha
    train_data_ratio = args.train_data_ratio
    wrong_label_ratio = args.wrong_label_ratio
    beta = args.beta
    lamda = args.lamda
    ct_prompt=args.ct_prompt
    ct_llm=args.ct_llm
    ct_ratio=args.ct_ratio
    compression_style=args.compression_style


    print(f"task={task}")
    result_log_dir="results/"+task+"_result.txt"
    print(f"result_log_dir={result_log_dir}")
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    json_file=open("config.json")
    configs = json.load(json_file)
    configs = {conf["name"] : conf for conf in configs}
    config = configs[task]

    tableA=config['tableA']
    tableB=config['tableB']

    tableB_ct = config['tableB_ct'] # data/amazon-google/tableB_ct.csv
    if ct_prompt:
        tableB_ct = tableB_ct.replace('.csv', f'_{ct_prompt}.csv')
        config['tableB_ct'] = tableB_ct
    print(f"tableB_ct={tableB_ct}")

    if os.path.exists(config['tableB_ct']):
        print(f"already have ct tableB...")
        pass 
    else:
        print(f"have no ct tableB, generating now...")
        ct_record_list = util.generate_counterfactual_records(tableB, ct_llm=ct_llm,ct_prompt=ct_prompt)
        util.save_counterfactual_records(ct_record_list, config['tableB_ct'])

    train=config['trainset']
    valid=config['validset']
    test=config['testset']

    serialized_entity_pairs_train_df=util.serialize_entity_pairs(tableA, tableB, train)
    print("train pairs count: ",len(serialized_entity_pairs_train_df))
    serialized_entity_pairs_train_ct_df=util.serialize_entity_pairs(tableA, tableB_ct, train)
    print("ct train pairs count: ",len(serialized_entity_pairs_train_ct_df))

    serialized_entity_pairs_valid_df=util.serialize_entity_pairs(tableA, tableB, valid)
    print("valid pairs count: ",len(serialized_entity_pairs_valid_df))
    serialized_entity_pairs_test_df=util.serialize_entity_pairs(tableA, tableB, test)
    print("test pairs count: ",len(serialized_entity_pairs_test_df))

    # train_df = pd.read_csv(train)
    # valid_df = pd.read_csv(valid)
    # test_df = pd.read_csv(test)

    emb_config_name='embedding_filedir_'+embedding_tool
    embedding_filedir=config[emb_config_name]

    if not os.path.exists(embedding_filedir):
        os.makedirs(embedding_filedir)

    embedding_filenameA = embedding_filedir + 'embA.txt'
    embedding_filenameB = embedding_filedir + 'embB.txt'
    embedding_filenameB_ct = embedding_filedir + 'embB_ct.txt'

    print(f"embedding_filenameA: {embedding_filenameA}")
    print(f"embedding_filenameB: {embedding_filenameB}")
    print(f"embedding_filenameB_ct: {embedding_filenameB_ct}")

    embeddingsA=util.load_or_generate_embeddings(tableA, embedding_filenameA,embedding_tool,prompt=emb_prompt)
    embeddingsB=util.load_or_generate_embeddings(tableB, embedding_filenameB,embedding_tool,prompt=emb_prompt)
    embeddingsB_ct=util.load_or_generate_embeddings(tableB_ct, embedding_filenameB_ct, embedding_tool,prompt=emb_prompt) 

    print(embeddingsA.shape)
    print(embeddingsB.shape)
    print(embeddingsB_ct.shape)

    """
        torch.Size([1081, 1024])
        torch.Size([1092, 1024])
        torch.Size([1091, 1024])
    """

    llm_dim = embeddingsA.shape[1]
    print('llm_dim:',llm_dim)

    tokenizer = transformers.AutoTokenizer.from_pretrained(slm)

    model = util.EntityPairEncoder(slm).to(device)
    slm_dim = model.slm_dim
    best_checkpoint_model = model.state_dict()
    matcher = util.Matcher(slm_dim).to(device)
    best_checkpoint_matcher = matcher.state_dict()
    autoencoder=util.AutoEncoder(llm_dim,slm_dim,hidden_dim).to(device)


    train_dataset = util.EntityPairDataset(tokenizer, serialized_entity_pairs_train_df, 
                                           embeddingsA, embeddingsB, serialized_entity_pairs_train_ct_df,
                                           embeddingsB_ct, label_reverse_ratio=wrong_label_ratio,is_ct=True,ct_ratio=ct_ratio)
    num_samples = int(len(train_dataset) * train_data_ratio)
    indices = list(range(num_samples))
    train_dataset = Subset(train_dataset, indices)

    valid_dataset = util.EntityPairDataset(tokenizer, serialized_entity_pairs_valid_df, embeddingsA, embeddingsB)
    test_dataset = util.EntityPairDataset(tokenizer, serialized_entity_pairs_test_df, embeddingsA, embeddingsB)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # print("test dataloader...")
    # for batch in train_dataloader:
    #     print(batch[0]['input_ids'].shape)
    #     print(batch[0]['attention_mask'].shape)
    #     print(batch[1].shape)
    #     print(batch[2])
    #     break
    # exit()

    compute_cse_loss = nn.CrossEntropyLoss()
    compute_recon_loss = nn.MSELoss()
    if compression_style=='kd':
        compute_align_loss = nn.KLDivLoss() # input target
        llm_matcher=nn.Linear(3072, 2).to(device) # text embedding 3 large dim=3072
        optimizer = optim.AdamW(list(model.parameters()) + list(matcher.parameters()) + list(llm_matcher.parameters()), lr=lr)

    else:
        compute_align_loss = nn.MSELoss()
        optimizer = optim.AdamW(list(model.parameters()) + list(matcher.parameters()) + list(autoencoder.parameters()), lr=lr)

    print("start training...")

    for epoch in trange(num_epochs):
        print(f"Training epoch {epoch}/{num_epochs}")
        model.train()
        matcher.train()
        epoch_loss_train = 0.0
        epoch_loss_valid = 0.0
        batch_count = 0
        for batch_idx, batch in enumerate(train_dataloader):
            idA, idB, inputs, llm_simi_emb, llm_simi_emb_ct, labels = batch['id_A'], batch['id_B'], batch['tokenized_pair'], batch['element_wise_product'].to(device),batch['element_wise_product_ct'].to(device), batch['label'].to(device)
            optimizer.zero_grad()
            input_ids = inputs['input_ids'].squeeze(1).to(device)  
            attention_mask = inputs['attention_mask'].squeeze(1).to(device)  
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            slm_dim=outputs.shape[-1]
            encoded,decoded=autoencoder(llm_simi_emb)
            encoded_ct,decoded_ct=autoencoder(llm_simi_emb_ct)
            if compression_style=='autoencoder':
                pass
            elif compression_style=='first':
                encoded = llm_simi_emb[:, :slm_dim]
                beta=0
            elif compression_style=='last':
                encoded = llm_simi_emb[:, slm_dim:]
                beta=0
            elif compression_style=='random':
                global random_indices
                if 'random_indices' not in globals():
                    random_indices = torch.randperm(llm_simi_emb.shape[1])[:slm_dim]
                encoded = llm_simi_emb[:, random_indices]
                beta=0
            elif compression_style=='kd':
                llm_simi_emb = llm_simi_emb.to(device)
                llm_matcher_outputs = llm_matcher(llm_simi_emb)
                matcher_outputs = matcher(outputs)
            else:
                raise exception('...')
            matcher_outputs = matcher(outputs)
            matching_loss = compute_cse_loss(matcher_outputs, labels)
            if compression_style=='kd':
                align_loss=compute_align_loss(matcher_outputs,llm_matcher_outputs)
                loss = matching_loss+alpha*align_loss
            else:
                recon_loss=compute_recon_loss(decoded, llm_simi_emb)
                align_loss=compute_align_loss(encoded,outputs)
                difenh=(1-F.cosine_similarity(encoded,encoded_ct, dim=1)) * lamda
                difenh = difenh.mean()
                loss = matching_loss+alpha*align_loss+beta*(recon_loss+difenh)
            loss.backward()
            optimizer.step()
            epoch_loss_train += loss.item()
            batch_count += 1
        average_epoch_loss = epoch_loss_train / batch_count
        print(f"Average training loss for epoch {epoch}: {average_epoch_loss:.4f}")

        # evaluation
        if epoch % epoch_count_for_valid == 0:
            model.eval()
            matcher.eval()
            output_list = []
            label_list = []
            with torch.no_grad():
                for batch in valid_dataloader: 
                    idA, idB, inputs, llm_simi_emb, labels = batch['id_A'], batch['id_B'], batch['tokenized_pair'], batch['element_wise_product'].to(device), batch['label'].to(device)
                    input_ids = inputs['input_ids'].squeeze(1).to(device)
                    attention_mask = inputs['attention_mask'].squeeze(1).to(device)
                    import time
                    
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    matcher_outputs = matcher(outputs) 
                    matcher_outputs_softmax = F.softmax(matcher_outputs, dim=1)[:, 1] 
                    
                    # labels = labels.long() 
                    matching_loss = compute_cse_loss(matcher_outputs, labels)
                    epoch_loss_valid += matching_loss.item()
                    output_list.append(matcher_outputs_softmax)
                    label_list.append(labels)

            output_list = torch.cat(output_list)
            label_list = torch.cat(label_list)
            all_labels = label_list.cpu().numpy()
            average_epoch_loss_valid = epoch_loss_valid / batch_count
            print(f"Average validation loss for epoch {epoch}: {average_epoch_loss_valid:.4f}")
            best_th = 0.0
            best_f1 = 0.0
            best_checkpoint = None
            for th in np.arange(0, 1.0, 0.01): 
                onehot_output_list = []
                for output, label in zip(output_list, label_list):
                    onehot_output_list.append(1 if output.item() > th else 0) 

                all_outputs = np.array(onehot_output_list)

                if len(all_labels) != len(all_outputs):
                    raise ValueError("")

                current_f1 = f1_score(all_labels, all_outputs)
                if current_f1 > best_f1:
                    best_f1 = current_f1
                    best_th = th
                    best_checkpoint_model = model.state_dict()
                    best_checkpoint_matcher = matcher.state_dict()
            print(f"Best threshold: {best_th}, Best validation F1 score: {best_f1}")

    print("training finished...")
    print("start testing...")
    model.load_state_dict(best_checkpoint_model)
    matcher.load_state_dict(best_checkpoint_matcher)
    model.eval()
    matcher.eval()
    output_list = []
    label_list = []
    with torch.no_grad():
        for batch in test_dataloader:
            idA, idB, inputs, llm_simi_emb, labels = batch['id_A'], batch['id_B'], batch['tokenized_pair'], batch['element_wise_product'].to(device), batch['label'].to(device)
            input_ids = inputs['input_ids'].squeeze(1).to(device)
            attention_mask = inputs['attention_mask'].squeeze(1).to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            matcher_outputs = F.softmax(matcher(outputs), dim=1)[:, 1]  
            output_list.append(matcher_outputs)
            label_list.append(labels)

    output_list = torch.cat(output_list)
    label_list = torch.cat(label_list)
    all_labels = label_list.cpu().numpy()
    
    best_th = 0.0
    best_f1 = 0.0
    best_precision = 0.0
    best_recall = 0.0
    for th in np.arange(0, 1.0, 0.001):  
        onehot_output_list = []
        for output, label in zip(output_list, label_list):
            if output.item() > th:
                onehot_output_list.append(1)
            else:
                onehot_output_list.append(0)
            
        all_outputs = np.array(onehot_output_list)  
        
        current_f1 = f1_score(all_labels, all_outputs)
        current_precision = precision_score(all_labels, all_outputs)
        current_recall = recall_score(all_labels, all_outputs)
        
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_precision = current_precision
            best_recall = current_recall
            best_th = th
    print(f"Best threshold: {best_th}, Best test F1 score: {best_f1}")
    print("testing finished...")
    with open(result_log_dir, 'a') as log_file:
        log_file.write(f"**************************\n")
        import datetime
        current_time = datetime.datetime.now()
        log_file.write(f"Current time: {current_time}\n")
        log_file.write(f"args: {args}\n")
        log_file.write(f"best_precision: {best_precision}\n")
        log_file.write(f"best_recall: {best_recall}\n")
        log_file.write(f"best_f1: {best_f1}\n")
    print(args)
    print('********************finished!!!********************')
    print('********************finished!!!********************')
    print('********************finished!!!********************')