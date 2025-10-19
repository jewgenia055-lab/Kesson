#форматирование жесткости

def format_to_latex(number, precision=1):
    """Форматирует число в научную нотацию для LaTeX"""
    sci_notation = f"{number:.{precision}e}"
    mantissa, exponent = sci_notation.split('e')
    exponent = int(exponent) 
    return fr"{mantissa}\times 10^{{{exponent}}}"