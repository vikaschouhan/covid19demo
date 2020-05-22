import tornado
import tornado.ioloop
import tornado.web
from   tornado.web import RequestHandler, Application, url
from   tornado.ioloop import IOLoop
from   tornado import gen
import sys
import os
import json
import operator
import argparse
import multipart
import operator
from   io import BytesIO
from   modules.utils import *

# Global database dir
database_dir = "./form_data"

############# Some boring stuff ################################
def jresp(status_code, pay_load={}, message=''):
    resp = json.dumps({
               'status'     : status_code,
               'response'   : pay_load,
               'message'    : message,
           })
    return resp
# enddef

def bad_request(message='Wrong json format'):
    return jresp(400, message=message)
# enddef

def decode_multiform_data(data):
    s_pat = data.split(b"\r")[0][2:]
    parts = multipart.MultipartParser(BytesIO(multipart.to_bytes(data)), s_pat).parts()
    d_d   = {}
    for part_t in parts:
        if part_t.content_type != '':
            d_d[part_t.name] = {
                    'isfile'        : True,
                    'filename'      : part_t.filename,
                    'name'          : part_t.name,
                    'file'          : part_t.file,
                    'size'          : part_t.size,
                }
        else:
            try:
                value = tornado.escape.json_decode(part_t.value)
            except:
                value = part_t.value
            # endtry
            d_d[part_t.name] = {
                    'isfile'        : False,
                    'name'          : part_t.name,
                    'value'         : value,
                }
        # endif
    # endfor
    return d_d
# enddef

############# Interesting stuff ################################
class BaseHandler0(RequestHandler):
    def set_default_headers(self):
        self.set_header("access-control-allow-origin", "*")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with")
        self.set_header('Access-Control-Allow-Methods', 'GET, PUT, POST, DELETE, OPTIONS')
        # HEADERS!
        self.set_header("Access-Control-Allow-Headers", "Access-Control-Allow-Origin,Authorization,Content-Type")
    # enddef

    def options(self):
        # no body
        self.set_status(204)
        self.finish()
# endclass

class MainPageHandler(BaseHandler0):
    async def post(self):
        form_data = decode_multiform_data(self.request.body)
        image = None
        audio = None
        form  = None

        if 'audio' in form_data:
            audio = form_data['audio']['file'].read()
        # endif
        # Read forms
        if 'formdata' in form_data:
            form = form_data['formdata']['value']
        # endif

        # Create new data
        new_data_dir = '{}/p-{}'.format(database_dir, u4())
        mkdir(new_data_dir)

        # Dump records
        if form:
            with open('{}/form.json'.format(new_data_dir), 'w') as j_out:
                json.dump(form, j_out)
            # endwith
        # endif
        if audio:
            with open('{}/audio.wav'.format(new_data_dir), 'wb') as a_out:
                a_out.write(audio)
            # endwith
        # endif
            
        self.write(jresp(status_code=200, pay_load={}, message='Status OK'))
        self.finish()
    # enddef
# endclass

#######################################################
# App routing control
def make_app(en_debug=False):
    app_handlers = [
            url(r'/submit_form', MainPageHandler),
        ]

    return Application(app_handlers, debug=en_debug)
# enddef

######################################################
# Main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file',  help='config file in json format.', type=str, default=None)
    args = parser.parse_args()

    if args.__dict__['config_file'] is None:
        print('--config_file is required !!')
        sys.exit(-1)
    # endif

    # Read config file
    config_file = rp(args.__dict__['config_file'])
    config_dict = load_json(config_file)

    if config_dict['num_processes'] > 1 and config_dict['enable_debug']:
        print('Server cant start in debug mode for multiprocessing mode.')
        sys.exit(-1)
    # endif

    # Initiate routing
    print('>> Starting to listen on port {}'.format(config_dict['server_port']))
    print('>> Spawing {} process{}.'.format(config_dict['num_processes'], '' if config_dict['num_processes'] < 2 else 'es'))
    app = make_app(config_dict['enable_debug'])
    server = tornado.httpserver.HTTPServer(app)
    server.bind(config_dict['server_port'])  # port
    server.start(config_dict['num_processes'])

    # Get ready
    print('>> Server ready.')
    tornado.ioloop.IOLoop.current().start()
# endif
