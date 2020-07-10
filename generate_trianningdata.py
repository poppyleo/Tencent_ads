import  pandas as pd
from config import Config
from tqdm import tqdm
import os
config = Config()
import csv
# flags = tf.flags
#
# FLAGS = flags.FLAGS
# flags.DEFINE_string("jump", 0,
#                     "跳长")

if __name__ == '__main__':
    """
    全部的列字典长度：76554
    去掉creatvie_id的列字典长度：44098
    去掉ad_id的列字典长度：44139
    """
    jump= 19
    start,end = (jump-1)*100000,(jump)*100000

    print('Jump:',jump)
    merge_df = pd.read_csv(config.data_processed+'merge_mapping.csv')
    label_df = pd.read_csv(config.data_path + 'train_preliminary/user.csv')
    merge_df = merge_df.merge(label_df,how='left',on='user_id')
    merge_df['age'].fillna(0, inplace=True)
    merge_df['gender'].fillna(0, inplace=True)
    # flags.mark_flag_as_required("jump")

    """根据训练集测试集"""
    #csv
    f1 = open(config.data_processed+ 'ad_finetune_{}_{}.csv'.format(start,end),'w')
    f2 = open(config.data_processed+ 'creative_finetune_{}_{}.csv'.format(start,end),'w')

    writer_1 = csv.writer(f1)
    writer_2 = csv.writer(f2)

    writer_1.writerow(["user_id", "text", "age",'gender'])
    writer_2.writerow(["user_id", "text", "age",'gender'])

    #txt
    c1 = open(config.data_processed + 'ad_corups_{}_{}.txt'.format(start, end), 'w') #txt
    c2 = open(config.data_processed + 'creative_corups_{}_{}.txt'.format(start, end), 'w')
    n = 0
    max_len = 0
    min_len = 1000000
    for user_id in tqdm(merge_df['user_id'].unique()[start:end]):  #190w个用户
        """为每一个用户生成特殊语料"""
        merge_user = merge_df[merge_df['user_id']==user_id]
        merge_sort = merge_user.sort_values('time') #按时间排序

        creative_id = list(merge_sort['creative_id'])  #creative
        ad_id = list(merge_sort['ad_id'])   #ad
        product_id = list(merge_sort['product_id'].unique())
        product_category = list(merge_sort['product_category'].unique())
        industry = list(merge_sort['industry'].unique())
        adertiser_id = list(merge_sort['advertiser_id'].unique())

        """文本格式"""
        creative_text = ' '.join([str(i) for i in creative_id if i!='\\N']) #丢掉空的
        ad_text = ' '.join([str(i) for i in ad_id if i!='\\N'] )
        product_text = ' '.join([str(i) for i in product_id if i!='\\N'])
        category_text = ' '.join([str(i) for i in product_category if i!='\\N'])
        industry_text = ' '.join([str(i) for i in industry if i!='\\N'])
        adertiser_text = ' '.join([str(i) for i in adertiser_id if i!='\\N'])
        write1_text = category_text + ' 天 '+industry_text+' 地 '+ product_text + ' 玄 '+ adertiser_text +' 黄 '+ad_text
        write2_text = category_text + ' 天 '+industry_text+' 地 '+ product_text + ' 玄 '+ adertiser_text +' 黄 '+creative_text


        writer_1.writerow([user_id,write1_text,int(merge_user['age'].iloc[0]),int(merge_user['gender'].iloc[0])])
        writer_2.writerow([user_id,write2_text,int(merge_user['age'].iloc[0]),int(merge_user['gender'].iloc[0])])

        c1.write(write1_text)
        c2.write(write2_text)
        c1.write('\n')
        c2.write('\n')

        """统计打印字长"""
        if n==0:
            avg_len = len(write2_text.split(' '))
        else:
            avg_len=(avg_len+len(write2_text.split(' ')))/2
        if len(write2_text.split(' '))>max_len:
            max_len = len(write2_text.split(' '))
        if len(write2_text.split(' '))<min_len:
            min_len=len(write2_text.split(' '))
        n+=1
        if n%1000==0:
            print('\n')
            print('序列平均长度为{}'.format(avg_len))
            print('序列最大长度为{}'.format(max_len))
            print('序列最小长度为{}'.format(min_len))
    c1.close()
    c2.close()

