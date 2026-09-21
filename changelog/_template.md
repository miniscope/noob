{% for section, _ in sections.items() %}
{% if section %}

### {{ section }}
{% endif %}
{% if sections[section] %}
{% for category, val in definitions.items() if category in sections[section] %}

**{{ definitions[category]['name'] }}**

{% for text, values in sections[section][category].items() %}
- {% if values %}{{ values|join(', ') }} - {% endif %}{{ text }}
{% endfor %}
{% endfor %}
{% endif %}
{% endfor %}
{{ "\n" }}
