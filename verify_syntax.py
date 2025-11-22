#!/usr/bin/env python3
"""
语法验证脚本 - 检查所有Python文件的语法正确性
"""

import py_compile
import os
import sys

def check_syntax(filepath):
    """检查单个文件的语法"""
    try:
        py_compile.compile(filepath, doraise=True)
        return True, None
    except py_compile.PyCompileError as e:
        return False, str(e)

def main():
    """主函数"""
    files_to_check = [
        'cfm_vc/models/encoder.py',
        'cfm_vc/models/decoder.py',
        'cfm_vc/models/context.py',
        'cfm_vc/models/flow.py',
        'cfm_vc/models/cfmvc.py',
        'cfm_vc/training/stage1_vae.py',
        'cfm_vc/training/stage2_flow.py',
    ]

    all_pass = True
    print("=" * 60)
    print("CFM-VC 代码语法验证")
    print("=" * 60)

    for filepath in files_to_check:
        full_path = os.path.join('/home/user/xuni2', filepath)
        if not os.path.exists(full_path):
            print(f"❌ {filepath}: 文件不存在")
            all_pass = False
            continue

        success, error = check_syntax(full_path)
        if success:
            print(f"✅ {filepath}: 语法正确")
        else:
            print(f"❌ {filepath}: 语法错误")
            print(f"   {error}")
            all_pass = False

    print("=" * 60)
    if all_pass:
        print("✅ 所有文件语法验证通过！")
        return 0
    else:
        print("❌ 存在语法错误，请检查上述文件")
        return 1

if __name__ == "__main__":
    sys.exit(main())
