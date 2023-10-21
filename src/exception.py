import sys
from src.logger import logging

def message_detail(error, error_detail:sys):
    _, _, exc = error_detail.exc_info()
    filename = exc.tb_frame.f_code.co_filename
    error_message = f"Error Occured - Python Script: [{filename}]  Line Number: [{exc.tb_lineno}] Error Message: [{error}]"

    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message
    

if __name__=="__main__":
    try:
        x = 1 + "1"
    except Exception as e:
        logging.warning(CustomException(e,sys))