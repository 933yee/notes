@echo off
echo hexo clean begin..
call cmd /c "hexo clean"

echo hexo generate begin..
call cmd /c "hexo generate"

git add .
git commit -m "update"
git subtree push --prefix=public origin gh-pages -f 