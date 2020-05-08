import requests
import PIL.Image
from   modules.utils import static

@static('url', None)
def base_url(url="http://ec2-34-216-83-21.us-west-2.compute.amazonaws.com:8000"):
    if base_url.url is None:
        base_url.url = url
        return url
    # endif

    return base_url.url
# enddef

def get_pred_endpoint():
    return base_url() + '/pred'
# enddef

def get_pred_b64_endpoint():
    return base_url() + '/predb64'
# enddef

def resolve_image_pred_to_sick_score(resp):
    if resp.status_code != 200:
        return 0.0
    # endif
    if resp.json()['status'] != 200:
        return 0.0
    # endif

    preds = resp.json()['response'][0]['scores']
    return max(preds.values())
# enddef

def predict_image_from_byte_stream(byte_stream):
    req_t = requests.post(get_pred_endpoint(), files={'image' : byte_stream})
    return resolve_image_pred_to_sick_score(req_t)
# enddef

def predict_image_from_file(image_file):
    return predict_image_from_byte_stream(open(image_file, 'rb'))
# enddef
