import os
import json
import pandas as pd
from sqlalchemy import create_engine
from typing import Callable

def write_output(result, df_result, prefix, output_dir, output_format, log_func: Callable = print, sql_db=None, skip_existing=False):
    """
    根据指定格式输出结果到文件，支持json/csv(excel)/sql/markdown。
    :param result: 原始结果（dict）
    :param df_result: DataFrame格式结果
    :param prefix: 文件名前缀
    :param output_dir: 输出文件夹
    :param output_format: 输出格式
    :param log_func: 日志函数
    :param sql_db: SQL数据库路径（可选）
    :param skip_existing: 是否跳过已存在的文件（默认False，即覆盖）
    """
    os.makedirs(output_dir, exist_ok=True)
    try:
        if output_format == 'json':
            out_path = os.path.join(output_dir, f'{prefix}_result.json')
            if skip_existing and os.path.exists(out_path):
                log_func(f'[Info] {out_path} 已存在，跳过生成')
            else:
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                log_func(f"[Info] 已保存JSON结果: {out_path}")
        elif output_format == 'dataframe':
            out_path = os.path.join(output_dir, f'{prefix}_result.csv')
            if skip_existing and os.path.exists(out_path):
                log_func(f'[Info] {out_path} 已存在，跳过生成')
            else:
                df_result.to_csv(out_path, index=False, encoding='utf-8')
                log_func(f"[Info] 已保存CSV结果: {out_path}")
        elif output_format == 'excel':
            out_path = os.path.join(output_dir, f'{prefix}_result.xlsx')
            if skip_existing and os.path.exists(out_path):
                log_func(f'[Info] {out_path} 已存在，跳过生成')
            else:
                df_result.to_excel(out_path, index=False)
                log_func(f"[Info] 已保存Excel结果: {out_path}")
        elif output_format == 'sql':
            if sql_db is None:
                sql_db = os.path.join(output_dir, 'result.db')
            table_name = f'{prefix}_result'
            engine = create_engine(f'sqlite:///{sql_db}')
            df_result.to_sql(table_name, engine, if_exists='replace', index=False)
            log_func(f"[Info] 已保存SQL结果: {sql_db} (表名: {table_name})")
        elif output_format == 'markdown':
            out_path = os.path.join(output_dir, f'{prefix}_result.md')
            if skip_existing and os.path.exists(out_path):
                log_func(f'[Info] {out_path} 已存在，跳过生成')
            else:
                with open(out_path, 'w', encoding='utf-8') as f:
                    f.write(df_result.to_markdown(index=False))
                log_func(f"[Info] 已保存Markdown结果: {out_path}")
        else:
            log_func(f"[Error] 不支持的输出格式: {output_format}")
    except Exception as e:
        log_func(f"[Error] 输出文件保存失败: {e}") 