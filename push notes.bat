@echo off
echo hexo clean begin..
call cmd /c "hexo clean"

echo hexo generate begin..
call cmd /c "hexo generate"

git add .
git commit -m "update"
git subtree split --prefix=public -b gh-pages
git push origin gh-pages --force