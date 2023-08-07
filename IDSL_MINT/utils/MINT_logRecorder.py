def MINT_logRecorder(messageQuote, logMINT = None, allowedPrinting = True, warning_message = False):
    
    if allowedPrinting:

        if warning_message:
            col = "91"
        else:
            col = "92"

        print(f"\033[0;{col}m{messageQuote}\033[0m")
 
    if logMINT is not None:
        try:
            with open(logMINT, 'a') as f:
                f.write(f"{messageQuote}\n")
        except:
            pass

    return None