import  pandas as pd
from config import Config
from tqdm import tqdm
import os
config = Config()
import tensorflow as tf

# flags = tf.flags
#
# FLAGS = flags.FLAGS
# flags.DEFINE_string("jump", 0,
#                     "跳长")

if __name__ == '__main__':
    """
    全部的列字典长度：76554
    去掉creatvie_id的列字典长度：48652
    去掉ad_id的列字典长度：48516
    """

    merge_df = pd.read_csv(config.data_processed+'merge_mapping.csv')

    # flags.mark_flag_as_required("jump")
    jump=19
    start,end = (jump-1)*100000,(jump)*100000
    """根据user_id生成语料"""
    f = open(config.data_path + 'corups_{}_{}.txt'.format(start,end),'w')
    n = 0
    max_len = 0
    min_len = 1000
    for user_id in tqdm(merge_df['user_id'].unique()[start:end]):  #190w个用户
        """为每一个用户生成特殊语料"""
        merge_user = merge_df[merge_df['user_id']==user_id]
        merge_sort = merge_user.sort_values('time') #按时间排序

        # creative_id = list(merge_sort['creative_id'])  #creative
        ad_id = list(merge_sort['ad_id'])   #ad
        product_id = list(merge_sort['product_id'])
        product_category = list(merge_sort['product_category'].unique())
        industry = list(merge_sort['industry'].unique())
        adertiser_id = list(merge_sort['advertiser_id'])
        click_time = list(merge_sort['click_times'])
        """文本格式"""
        # creative_text = ' '.join([str(i) for i in creative_id if i!='\\N']) #丢掉空的
        ad_text = ' '.join([str(i) for i in ad_id if i!='\\N'] )
        product_text = ' '.join([str(i) for i in product_id if i!='\\N'])
        category_text = ' '.join([str(i) for i in product_category if i!='\\N'])
        industry_text = ' '.join([str(i) for i in industry if i!='\\N'])
        adertiser_text = ' '.join([str(i) for i in adertiser_id if i!='\\N'])
        click_text = ' '.join([str(i) for i in click_time if i!='\\N'])

        # write_text = creative_text +' -1 ' + product_text\
        #              + ' -1 '+ category_text + ' -1 '+industry_text+' -1 '+ adertiser_text +' -1 '+click_text
        write_text = ad_text +' 隔 ' + product_text\
                     + ' 隔 '+ category_text + ' 隔 '+industry_text+' 隔 '+ adertiser_text +' 隔 '+click_text
        # write_text = creative_text +' -1 '+ ad_text +' -1 ' + product_text\
        #              + ' -1 '+ category_text + ' -1 '+industry_text+' -1 '+ adertiser_text +' -1 '+click_text  #
        f.write(write_text)
        f.write('\n')
        """统计打印字长"""
        if n==0:
            avg_len = len(write_text.split(' '))
        else:
            avg_len=(avg_len+len(write_text.split(' ')))/2
        if len(write_text.split(' '))>max_len:
            max_len = len(write_text.split(' '))
        if len(write_text.split(' '))<min_len:
            min_len=len(write_text.split(' '))
        n+=1
        if n%10000==0:
            print('\n')
            print('序列平均长度为{}'.format(avg_len))
            print('序列最大长度为{}'.format(max_len))
            print('序列最小长度为{}'.format(min_len))
    f.close()

