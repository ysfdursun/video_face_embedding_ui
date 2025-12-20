import os
import urllib.request
import ssl
from django.core.management.base import BaseCommand
from django.conf import settings

class Command(BaseCommand):
    help = 'Downloads external static libraries (Tailwind, Flowbite) to local static directory'

    def handle(self, *args, **options):
        # Hedef klasör: core/static/core/vendor
        base_dir = settings.BASE_DIR
        target_dir = os.path.join(base_dir, 'core', 'static', 'core', 'vendor')
        
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            self.stdout.write(self.style.SUCCESS(f'Created directory: {target_dir}'))

        # İndirilecek dosyalar listesi
        libraries = [
            {
                'name': 'tailwindcss.js',
                'url': 'https://cdn.tailwindcss.com?plugins=forms,typography,aspect-ratio',
            },
            {
                'name': 'flowbite.min.css',
                'url': 'https://cdnjs.cloudflare.com/ajax/libs/flowbite/2.2.1/flowbite.min.css'
            },
            {
                'name': 'flowbite.min.js',
                'url': 'https://cdnjs.cloudflare.com/ajax/libs/flowbite/2.2.1/flowbite.min.js'
            },
            {
                'name': 'alpine.min.js',
                'url': 'https://cdn.jsdelivr.net/npm/alpinejs@3.13.3/dist/cdn.min.js'
            }
        ]

        self.stdout.write('Downloading assets...')
        
        # SSL context (cert verify failed hatasını önlemek için gerekebilir, basit scriptlerde)
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        for lib in libraries:
            file_path = os.path.join(target_dir, lib['name'])
            try:
                self.stdout.write(f"Downloading {lib['name']}...")
                # urllib ile indir
                with urllib.request.urlopen(lib['url'], context=ctx, timeout=15) as response:
                    content = response.read()
                    with open(file_path, 'wb') as f:
                        f.write(content)
                
                self.stdout.write(self.style.SUCCESS(f"✓ Saved to {file_path}"))
                
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"✗ Failed to download {lib['name']}: {e}"))

        self.stdout.write(self.style.SUCCESS('\nAll assets downloaded successfully.'))
