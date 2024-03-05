curl -X POST https://apps-dev.inside.anl.gov/gottextai/api/v1/extracttext \
     -F "file=@./Mine/MOE.pdf" \
     -F "clean_for_corpus=False" \
     -F "simple_clean_text=True" \
     -F "simple_summary=False"

     
