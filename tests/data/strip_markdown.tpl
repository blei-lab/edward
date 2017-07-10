{% extends 'python.tpl'%}
# Remove markdown cells.
{% block markdowncell -%}
{% endblock markdowncell %}

# Change the appearance of execution count.
{% block in_prompt %}
{%- endblock in_prompt %}
