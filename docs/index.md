---
layout: default
---

# VLSF

I am a researcher/developer... [your intro here]

### Research Interests
* 123
* 456

### MathJax
When $$a \ne 0$$, there are two solutions to $$ax^2 + bx + c = 0$$ and they are:
$$x = {-b \pm \sqrt{b^2-4ac} \over 2a}$$

### What's New

{% for post in site.posts limit:5 %}
* **{{ post.date | date: "%b %d, %Y" }}**: [{{ post.title }}]({{ post.url }})
{% endfor %}

[See all updates](/archive.md)

Check out my [Projects](notes/projects.md).
