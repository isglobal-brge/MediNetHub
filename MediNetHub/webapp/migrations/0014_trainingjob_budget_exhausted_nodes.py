from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('webapp', '0013_trainingjob_privacy_delta_trainingjob_privacy_epsilon'),
    ]

    operations = [
        migrations.AddField(
            model_name='trainingjob',
            name='budget_exhausted_nodes',
            field=models.JSONField(
                blank=True,
                default=list,
                help_text='Connections that were rejected due to an exhausted ε budget.',
                null=True,
            ),
        ),
    ]
