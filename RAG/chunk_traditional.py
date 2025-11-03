# ->str 是类型注解，表示返回的类型应当是str，function：将给定的文章读取成一个完整的字符串
def read_data() ->str:
#with 语句：用于上下文管理，自动处理资源的打开和关闭，即使发生异常也能确保文件正确关闭
#open() 函数参数：'data.txt'：要打开的文件名；'r'：打开模式，r 表示只读；encoding='utf-8'：指定文件编码为 UTF-8
    with open('./data/GB+34668-2024-data.txt', 'r', encoding='utf-8') as f:
        # 读取文件内容并返回字符串类型
        return f.read()    

# 分块函数
def chunk_text() ->list[str]:
    # 调用 read_data 函数获取文章内容, 变量类型注解
    content: str = read_data()
    # 按照换行符将文章内容分割成多个段落，返回一个列表
    chunks: list[str] = content.split('\n\n')   

    result : list[any] = []
    for para in chunks:
        if len(para.strip()) > 0:  # 去除空段落, .strip()方法用于移除字符串首尾的空白字符（空格、换行符、制表符等）
            result.append(para.strip())

    return result

if __name__ == '__main__':
    # 调用 chunk_text 函数获取分块后的段落列表
    paragraphs = chunk_text()
    # 遍历每个段落并打印
    for para in paragraphs:
        print(para)
        print("---------------------------")
