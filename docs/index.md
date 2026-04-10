---
layout: main
---

# Vladimir Fanaskov

Independent researcher.

### Research Interests

* numerical analysis
* scientific computing

### What's New

{% for post in site.posts limit:5 %}
* **{{ post.date | date: "%b %d, %Y" }}**: [{{ post.title }}]({{ post.url }})
{% endfor %}

[See all updates](/archive.md)

Check out my [Projects](notes/projects.md).
