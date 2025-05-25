def process(train_raw,
            valid_raw,
            test_raw,
            table_A_csv,
            table_B_csv,
            train_csv,
            valid_csv,
            test_csv
            ):
    
    # 检查并创建目录
    import os
    output_dir = os.path.dirname(table_A_csv)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    def read_file(file_path):
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(line.strip().split('\t'))
        return data
        
    train_data = read_file(train_raw)
    valid_data = read_file(valid_raw)
    test_data = read_file(test_raw)

    print("训练集大小:", len(train_data), "行", len(train_data[0]) if train_data else 0, "列")
    print("验证集大小:", len(valid_data), "行", len(valid_data[0]) if valid_data else 0, "列") 
    print("测试集大小:", len(test_data), "行", len(test_data[0]) if test_data else 0, "列")

    # 从第一行数据中提取列名
    first_record = train_data[0][0]  # 获取第一行的r1
    col_names = []
    
    # 按COL和VAL分割并提取列名
    parts = first_record.split('COL')
    for part in parts[1:]:  # 跳过第一个空字符串
        if 'VAL' in part:
            col_name = part.split('VAL')[0].strip()
            col_names.append(col_name)
            
    print("提取的列名:", col_names)
    # 初始化六个列表
    train_tableA_list = []
    train_tableB_list = []
    valid_tableA_list = []
    valid_tableB_list = []
    test_tableA_list = []
    test_tableB_list = []

    # 处理训练数据
    for record in train_data:
        # 每条记录包含三列数据
        tableA = record[0]  # 第一列是tableA的记录
        tableB = record[1]  # 第二列是tableB的记录
        label = int(record[2])  # 第三列是标签,转换为整数
        
        train_tableA_list.append(tableA)
        train_tableB_list.append(tableB)

    # 处理验证数据
    for record in valid_data:
        tableA = record[0]
        tableB = record[1]
        
        valid_tableA_list.append(tableA)
        valid_tableB_list.append(tableB)

    # 处理测试数据
    for record in test_data:
        tableA = record[0]
        tableB = record[1]
        
        test_tableA_list.append(tableA)
        test_tableB_list.append(tableB)

    print("训练集:")
    print("tableA_list大小:", len(train_tableA_list))
    print("tableB_list大小:", len(train_tableB_list))
    print("\n验证集:")
    print("tableA_list大小:", len(valid_tableA_list))
    print("tableB_list大小:", len(valid_tableB_list))
    print("\n测试集:")
    print("tableA_list大小:", len(test_tableA_list))
    print("tableB_list大小:", len(test_tableB_list))

    # 初始化存储拆分后值的列表
    train_tableA_values = []
    train_tableB_values = []
    valid_tableA_values = []
    valid_tableB_values = []
    test_tableA_values = []
    test_tableB_values = []

    # 处理训练集tableA_list
    for record in train_tableA_list:
        values = []
        parts = record.split('COL')[1:]  # 跳过第一个空字符串
        for part in parts:
            if 'VAL' in part:
                value = part.split('VAL')[1].strip()
                values.append(value)
        train_tableA_values.append(values)

    # 处理训练集tableB_list  
    for record in train_tableB_list:
        values = []
        parts = record.split('COL')[1:]  # 跳过第一个空字符串
        for part in parts:
            if 'VAL' in part:
                value = part.split('VAL')[1].strip()
                values.append(value)
        train_tableB_values.append(values)

    # 处理验证集tableA_list
    for record in valid_tableA_list:
        values = []
        parts = record.split('COL')[1:]  # 跳过第一个空字符串
        for part in parts:
            if 'VAL' in part:
                value = part.split('VAL')[1].strip()
                values.append(value)
        valid_tableA_values.append(values)

    # 处理验证集tableB_list
    for record in valid_tableB_list:
        values = []
        parts = record.split('COL')[1:]  # 跳过第一个空字符串
        for part in parts:
            if 'VAL' in part:
                value = part.split('VAL')[1].strip()
                values.append(value)
        valid_tableB_values.append(values)

    # 处理测试集tableA_list
    for record in test_tableA_list:
        values = []
        parts = record.split('COL')[1:]  # 跳过第一个空字符串
        for part in parts:
            if 'VAL' in part:
                value = part.split('VAL')[1].strip()
                values.append(value)
        test_tableA_values.append(values)

    # 处理测试集tableB_list
    for record in test_tableB_list:
        values = []
        parts = record.split('COL')[1:]  # 跳过第一个空字符串
        for part in parts:
            if 'VAL' in part:
                value = part.split('VAL')[1].strip()
                values.append(value)
        test_tableB_values.append(values)

    print("训练集:")
    print("tableA_values第一行示例:", train_tableA_values[0])
    print("tableB_values第一行示例:", train_tableB_values[0])
    print("tableA_values大小:", len(train_tableA_values))
    print("tableB_values大小:", len(train_tableB_values))
    
    print("\n验证集:")
    print("tableA_values第一行示例:", valid_tableA_values[0])
    print("tableB_values第一行示例:", valid_tableB_values[0]) 
    print("tableA_values大小:", len(valid_tableA_values))
    print("tableB_values大小:", len(valid_tableB_values))

    print("\n测试集:")
    print("tableA_values第一行示例:", test_tableA_values[0])
    print("tableB_values第一行示例:", test_tableB_values[0])
    print("tableA_values大小:", len(test_tableA_values))
    print("tableB_values大小:", len(test_tableB_values))

    # 将列表转换为pandas DataFrame
    import pandas as pd
    
    # 创建训练集的DataFrame
    train_df_tableA = pd.DataFrame(train_tableA_values, columns=col_names)
    train_df_tableB = pd.DataFrame(train_tableB_values, columns=col_names)
    
    # 创建验证集的DataFrame
    valid_df_tableA = pd.DataFrame(valid_tableA_values, columns=col_names)
    valid_df_tableB = pd.DataFrame(valid_tableB_values, columns=col_names)
    
    # 创建测试集的DataFrame
    test_df_tableA = pd.DataFrame(test_tableA_values, columns=col_names)
    test_df_tableB = pd.DataFrame(test_tableB_values, columns=col_names)
    
    print("训练集:")
    print("tableA DataFrame的形状:", train_df_tableA.shape)
    print("tableB DataFrame的形状:", train_df_tableB.shape)
    print("\ntableA DataFrame的前几行:")
    print(train_df_tableA.head())
    print("\ntableB DataFrame的前几行:")
    print(train_df_tableB.head())
    
    print("\n验证集:")
    print("tableA DataFrame的形状:", valid_df_tableA.shape) 
    print("tableB DataFrame的形状:", valid_df_tableB.shape)
    print("\ntableA DataFrame的前几行:")
    print(valid_df_tableA.head())
    print("\ntableB DataFrame的前几行:")
    print(valid_df_tableB.head())
    
    print("\n测试集:")
    print("tableA DataFrame的形状:", test_df_tableA.shape)
    print("tableB DataFrame的形状:", test_df_tableB.shape)
    print("\ntableA DataFrame的前几行:")
    print(test_df_tableA.head())
    print("\ntableB DataFrame的前几行:") 
    print(test_df_tableB.head())

    # 为所有DataFrame添加id列,对相同record使用相同id
    # 处理tableA的id
    all_tableA = pd.concat([train_df_tableA, valid_df_tableA, test_df_tableA])
    record_to_id_A = {}
    current_id = 0
    for _, row in all_tableA.iterrows():
        # 只使用原始列创建元组
        row_tuple = tuple(row[col_names])
        if row_tuple not in record_to_id_A:
            record_to_id_A[row_tuple] = current_id
            current_id += 1
    
    train_df_tableA['id'] = [record_to_id_A[tuple(row[col_names])] for _, row in train_df_tableA.iterrows()]
    valid_df_tableA['id'] = [record_to_id_A[tuple(row[col_names])] for _, row in valid_df_tableA.iterrows()]
    test_df_tableA['id'] = [record_to_id_A[tuple(row[col_names])] for _, row in test_df_tableA.iterrows()]
    
    # 处理tableB的id
    all_tableB = pd.concat([train_df_tableB, valid_df_tableB, test_df_tableB])
    record_to_id_B = {}
    current_id = 0
    for _, row in all_tableB.iterrows():
        # 只使用原始列创建元组
        row_tuple = tuple(row[col_names])
        if row_tuple not in record_to_id_B:
            record_to_id_B[row_tuple] = current_id
            current_id += 1
    
    train_df_tableB['id'] = [record_to_id_B[tuple(row[col_names])] for _, row in train_df_tableB.iterrows()]
    valid_df_tableB['id'] = [record_to_id_B[tuple(row[col_names])] for _, row in valid_df_tableB.iterrows()]
    test_df_tableB['id'] = [record_to_id_B[tuple(row[col_names])] for _, row in test_df_tableB.iterrows()]

    # 合并所有tableA数据并保存
    all_tableA = pd.concat([train_df_tableA, valid_df_tableA, test_df_tableA])
    all_tableA = all_tableA[['id'] + col_names]  # 重新排列列,使id在最前
    all_tableA.to_csv(table_A_csv, index=False)
    print(f"\n已保存所有tableA数据到 {table_A_csv}")
    print("tableA的形状:", all_tableA.shape)
    
    # 合并所有tableB数据并保存  
    all_tableB = pd.concat([train_df_tableB, valid_df_tableB, test_df_tableB])
    all_tableB = all_tableB[['id'] + col_names]  # 重新排列列,使id在最前
    all_tableB.to_csv(table_B_csv, index=False)
    print(f"已保存所有tableB数据到 {table_B_csv}")
    print("tableB的形状:", all_tableB.shape)
    
    # 创建训练集CSV
    train_pairs = []
    for i in range(len(train_data)):
        ltable_id = record_to_id_A[tuple(train_df_tableA.iloc[i][col_names])]
        rtable_id = record_to_id_B[tuple(train_df_tableB.iloc[i][col_names])]
        label = int(train_data[i][2])
        train_pairs.append([ltable_id, rtable_id, label])
    
    train_df = pd.DataFrame(train_pairs, columns=['ltable_id', 'rtable_id', 'label'])
    train_df.to_csv(train_csv, index=False)
    print(f"\n已保存训练集到 {train_csv}")
    print("训练集的形状:", train_df.shape)

    # 创建验证集CSV 
    valid_pairs = []
    for i in range(len(valid_data)):
        ltable_id = record_to_id_A[tuple(valid_df_tableA.iloc[i][col_names])]
        rtable_id = record_to_id_B[tuple(valid_df_tableB.iloc[i][col_names])]
        label = int(valid_data[i][2])
        valid_pairs.append([ltable_id, rtable_id, label])
    
    valid_df = pd.DataFrame(valid_pairs, columns=['ltable_id', 'rtable_id', 'label'])
    valid_df.to_csv(valid_csv, index=False)
    print(f"已保存验证集到 {valid_csv}")
    print("验证集的形状:", valid_df.shape)

    # 创建测试集CSV
    test_pairs = []
    for i in range(len(test_data)):
        ltable_id = record_to_id_A[tuple(test_df_tableA.iloc[i][col_names])]
        rtable_id = record_to_id_B[tuple(test_df_tableB.iloc[i][col_names])]
        label = int(test_data[i][2])
        test_pairs.append([ltable_id, rtable_id, label])
    
    test_df = pd.DataFrame(test_pairs, columns=['ltable_id', 'rtable_id', 'label'])
    test_df.to_csv(test_csv, index=False)
    print(f"已保存测试集到 {test_csv}")
    print("测试集的形状:", test_df.shape)
    

    

if __name__ == "__main__":
    # # 50 unseen
    # train_raw="data/wdc_20cc_raw/preprocessed_wdcproducts20cc80rnd000un_train_small.txt"
    # valid_raw="data/wdc_20cc_raw/preprocessed_wdcproducts20cc80rnd000un_valid_small.txt"
    # test_raw= "data/wdc_20cc_raw/preprocessed_wdcproducts20cc80rnd050un_gs.txt"
    
    # table_A=  "data/wdc_20cc_small_50un/tableA.csv"
    # table_B=  "data/wdc_20cc_small_50un/tableB.csv"
    # train_csv="data/wdc_20cc_small_50un/train.csv"
    # valid_csv="data/wdc_20cc_small_50un/valid.csv"
    # test_csv= "data/wdc_20cc_small_50un/test.csv"

    # 100 unseen
    train_raw="data/wdc_20cc_raw/preprocessed_wdcproducts20cc80rnd000un_train_small.txt"
    valid_raw="data/wdc_20cc_raw/preprocessed_wdcproducts20cc80rnd000un_valid_small.txt"
    test_raw= "data/wdc_20cc_raw/preprocessed_wdcproducts20cc80rnd100un_gs.txt"
    
    table_A=  "data/wdc_20cc_small_100un/tableA.csv"
    table_B=  "data/wdc_20cc_small_100un/tableB.csv"
    train_csv="data/wdc_20cc_small_100un/train.csv"
    valid_csv="data/wdc_20cc_small_100un/valid.csv"
    test_csv= "data/wdc_20cc_small_100un/test.csv"

    process(train_raw=train_raw,
           valid_raw=valid_raw,
           test_raw=test_raw,
           table_A_csv=table_A,
           table_B_csv=table_B,
           train_csv=train_csv,
           valid_csv=valid_csv,
           test_csv=test_csv)