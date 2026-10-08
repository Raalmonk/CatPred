"""File identity and certification checks only; no torch/model execution."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from catpred_accel import protein_loading as loading


class LoaderMetadataTests(unittest.TestCase):
    def test_import_does_not_load_torch_or_esm(self):
        command='import sys; import catpred_accel.protein_loading; assert "torch" not in sys.modules; assert "esm" not in sys.modules'
        subprocess.run([sys.executable,'-c',command],check=True)

    def test_file_identity_is_memoized_and_revalidated_after_change(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'checkpoint.pt';path.write_bytes(b'original')
            before=dict(loading._HASH_COUNTS)
            first=loading._sha(path);stamp=path.stat()
            second=loading._sha(path)
            self.assertEqual(first,second)
            self.assertEqual(loading._HASH_COUNTS['files_hashed']-before['files_hashed'],1)
            self.assertEqual(loading._HASH_COUNTS['cache_hits']-before['cache_hits'],1)
            # Same size and restored mtime are insufficient to reuse the digest:
            # inode/ctime are part of the cache identity as well.
            replacement=Path(directory)/'replacement';replacement.write_bytes(b'changed!')
            os.utime(replacement,ns=(stamp.st_atime_ns,stamp.st_mtime_ns));replacement.replace(path)
            self.assertNotEqual(loading._sha(path),first)
            self.assertEqual(loading._HASH_COUNTS['files_hashed']-before['files_hashed'],2)

    def test_loader_certificate_requires_matching_kind_mode_and_warm_evidence(self):
        key='a'*64
        record=dict(kind='esm_loader',mode='meta',context_key=key,passed=True,exact_bits=True,
                    warm_nonregression=True,evidence_sha256='b'*64)
        self.assertEqual(loading._certificate(key,[record]),record)
        self.assertIsNone(loading._certificate('c'*64,[record]))
        for field,value in [('kind','esm_features'),('mode','off'),('passed',False),('exact_bits',False),
                            ('warm_nonregression',False),('evidence_sha256','')]:
            self.assertIsNone(loading._certificate(key,[dict(record,**{field:value})]),field)


if __name__=='__main__':unittest.main()
