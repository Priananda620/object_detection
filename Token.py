import time
import hashlib
import os
import config


class FlusonicToken:
    def __init__(self, cctv_flusonic, lifetime_hr):
        self.KEY = config.FLUSONIC_TOKEN
        self.CCTV_FLUSONIC = cctv_flusonic
        self.LIFETIME = 3600 * (24*lifetime_hr);
    
    def get_tokenized_url(self):
        key = self.KEY
        lifetime = self.LIFETIME
        stream = self.CCTV_FLUSONIC
        
        desync = 300
        start_time = int(time.time()) - desync
        end_time = start_time + lifetime
        salt = os.urandom(16).hex()
        hash_str = stream + str(start_time) + str(end_time) + key + salt
        token = hashlib.sha1(hash_str.encode()).hexdigest() + '-' + salt + '-' + str(end_time) + '-' + str(start_time)
        url = 'https://cctv.molecool.id/' + stream + '/video.m3u8?token=' + token

        return url
